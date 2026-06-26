// NOTE(amg) An LLM generated this from my spec with a few rounds of feedback.
// By visual inspection this all looks correct to me, but more testing is probably warranted
// Tensor is intentionally using the unified memory right now. If this is a bottleneck 
// we should be able to switch to manually managed memory fairly easily later. The TensorView
// really does not care. You can also make a TensorView from any raw pointer, not just a Tensor.
// It is somewhat unclear to me if we actually want to allow implcit conversion of TensorView to T*
// Nice style-wise, but sort of a footgun

#pragma once

//#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <type_traits>

// ---------------------------------------------------------------------------
// Helper: checked CUDA call (host-only — throws std::runtime_error)
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error at ")             \
                + __FILE__ + ":" + std::to_string(__LINE__)                    \
                + " — " + cudaGetErrorString(_e));                             \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// TensorView<T, dim>
//
//  Non-owning, trivially-copyable view over a strided block of T.
//  Safe to pass by value into CUDA kernels.
//
//  All accessor logic lives here exactly once; Tensor<T,dim> inherits it.
//
//  Public interface:
//    data()              — raw pointer to first element
//    shape(i)            — size along dimension i
//    stride(i)           — stride along dimension i (elements, not bytes)
//    numel()             — total element count
//    operator()(i,j,...) — element access, exactly `dim` indices
//    slice<d>(index)     — fix dimension d, return TensorView<T, dim-1>
// ---------------------------------------------------------------------------
template <typename T, int dim>
struct TensorView {
    static_assert(dim >= 1, "dim must be >= 1.");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable for safe device access.");

    // Public data members — plain arrays keep the struct trivially copyable
    // even after Tensor inherits from it (no virtual, no user destructor here).
    T*  data_;
    int shape_[dim];
    int strides_[dim];  // in elements

    // ------------------------------------------------------------------
    // Default constructor — leaves fields uninitialised (matches old
    // aggregate behaviour; Tensor's constructor fills them in).
    // ------------------------------------------------------------------
    TensorView() = default;

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    __host__ __device__ T*       data()        { return data_; }
    __host__ __device__ const T* data()  const { return data_; }
    __host__ __device__ int      shape (int i) const { return shape_[i];   }
    __host__ __device__ int      stride(int i) const { return strides_[i]; }

    __host__ __device__ size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < dim; ++i) n *= static_cast<size_t>(shape_[i]);
        return n;
    }

    // ------------------------------------------------------------------
    // Element access — exactly `dim` indices required at compile time
    // ------------------------------------------------------------------

    template <typename... Idx,
              typename = std::enable_if_t<sizeof...(Idx) == dim>>
    __host__ __device__ T& operator()(Idx... idx) {
        return data_[linear_index(idx...)];
    }

    template <typename... Idx,
              typename = std::enable_if_t<sizeof...(Idx) == dim>>
    __host__ __device__ const T& operator()(Idx... idx) const {
        return data_[linear_index(idx...)];
    }

    // ------------------------------------------------------------------
    // Slicing — fix dimension `fixed_dim` at `index`.
    // Returns a TensorView<T, dim-1> (one fewer dimension).
    //
    // Usage:
    //   auto img = batch_tensor.slice<0>(2);   // fix batch dim → 2-D view
    //   auto row = img.slice<0>(5);            // fix row dim  → 1-D view
    // ------------------------------------------------------------------
    template <int fixed_dim>
    __host__ __device__ TensorView<T, dim - 1> slice(int index) const {
        static_assert(dim > 1,
            "Cannot slice a 1-D view — use operator()(i) instead.");
        static_assert(fixed_dim >= 0 && fixed_dim < dim,
            "fixed_dim out of range.");

        TensorView<T, dim - 1> v;
        v.data_ = data_ + index * strides_[fixed_dim];
        int out = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == fixed_dim) continue;
            v.shape_[out]   = shape_[i];
            v.strides_[out] = strides_[i];
            ++out;
        }
        return v;
    }


    // add dummy outter dimension of 1
    __host__ __device__ TensorView<T, dim + 1> unsqueeze() const {
        TensorView<T, dim + 1> v;
        v.data_ = data_;
        v.shape_[0] = 1;
        v.strides_[0] = numel();
        for (int i = 0; i < dim; ++i) {
            v.shape_[i + 1]   = shape_[i];
            v.strides_[i + 1] = strides_[i];
        }
        return v;
    }


    // ------------------------------------------------------------------
    // slice_1d — fix all dimensions except `kept_dim`, return a 1-D view.
    //
    // Template parameter:
    //   kept_dim — the dimension that becomes the axis of the 1-D view.
    //
    // Variadic arguments:
    //   exactly (dim - 1) indices, one per non-kept dimension, in
    //   ascending dimension order (skipping kept_dim).
    //
    // Usage (3-D tensor, shape [B, H, W]):
    //   auto col = t.slice_1d<1>(b, w);  // keep H → View1D of length H
    //   auto row = t.slice_1d<2>(b, h);  // keep W → View1D of length W
    //   auto bat = t.slice_1d<0>(h, w);  // keep B → View1D of length B
    // ------------------------------------------------------------------
    template <int kept_dim, typename... Idx,
              typename = std::enable_if_t<sizeof...(Idx) == dim - 1>>
    __host__ __device__ TensorView<T, 1> slice_1d(Idx... fixed_indices) const {
        static_assert(dim > 1,
            "slice_1d requires at least a 2-D view.");
        static_assert(kept_dim >= 0 && kept_dim < dim,
            "kept_dim out of range.");

        // Unpack the (dim-1) fixed indices into a plain array,
        // in order of dimension index with kept_dim skipped.
        int fixed[dim - 1] = {static_cast<int>(fixed_indices)...};

        // Walk all dims; accumulate pointer offset for fixed ones.
        T* ptr = data_;
        int fi = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == kept_dim) continue;
            ptr += fixed[fi++] * strides_[i];
        }

        TensorView<T, 1> v;
        v.data_      = ptr;
        v.shape_[0]   = shape_[kept_dim];
        v.strides_[0] = strides_[kept_dim];
        return v;
    }

    // ------------------------------------------------------------------
    // linear_index — shared by operator() in this class and Tensor below
    // ------------------------------------------------------------------
    template <typename... Idx>
    __host__ __device__ int linear_index(Idx... idx) const {
        int indices[dim] = {static_cast<int>(idx)...};
        int offset = 0;
        for (int i = 0; i < dim; ++i) offset += indices[i] * strides_[i];
        return offset;
    }


    __host__ __device__ void deep_copy_from(const TensorView<T, dim>& other) {
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            data_[i] = other.data_[i];
        }
    }

    // TODO(amg) might be a footgun to allow these implicit conversions, but makes the callsite look a bit nicer
    __host__ __device__ operator T*()             { return data_; }
    __host__ __device__ operator const T*() const { return data_; }


    //__host__ __device__ Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> as_eigen_vector() {
    //    static_assert(dim == 1, "as_eigen_vector only valid for 1-D views.");
    //    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(data_, shape_[0]);
    //}
};

// ---------------------------------------------------------------------------
// Tensor<T, dim>
//
//  Owning tensor backed by cudaMallocManaged.
//  Inherits all accessor / slice logic from TensorView<T, dim>.
//
//  Additional interface:
//    fill(value)     — host-side scalar fill
//    prefetch(dev)   — cudaMemPrefetchAsync (cudaCpuDeviceId for host)
//    sync()          — cudaDeviceSynchronize
//    view()          — explicit conversion to a bare TensorView by value
//                      (use this when passing to a kernel)
// ---------------------------------------------------------------------------
template <typename T, int dim>
class Tensor : public TensorView<T, dim> {
    using Base = TensorView<T, dim>;

public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    explicit Tensor(const std::array<int, dim>& shape) {
        for (int i = 0; i < dim; ++i) this->shape_[i] = shape[i];
        compute_strides();
        CUDA_CHECK(cudaMallocManaged(&this->data_, this->numel() * sizeof(T)));
    }

    template <typename... Dims,
              typename = std::enable_if_t<sizeof...(Dims) == dim>>
    explicit Tensor(Dims... dims)
        : Tensor(std::array<int, dim>{static_cast<int>(dims)...}) {}

    // ------------------------------------------------------------------
    // Ownership — non-copyable, moveable
    // ------------------------------------------------------------------

    ~Tensor() {
        if (this->data_) { cudaFree(this->data_); this->data_ = nullptr; }
    }

    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept {
        this->data_ = o.data_; o.data_ = nullptr;
        for (int i = 0; i < dim; ++i) {
            this->shape_[i]   = o.shape_[i];
            this->strides_[i] = o.strides_[i];
        }
    }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            if (this->data_) cudaFree(this->data_);
            this->data_ = o.data_; o.data_ = nullptr;
            for (int i = 0; i < dim; ++i) {
                this->shape_[i]   = o.shape_[i];
                this->strides_[i] = o.strides_[i];
            }
        }
        return *this;
    }

    // ------------------------------------------------------------------
    // Explicit view conversion — use when passing to a CUDA kernel.
    // Returns the base TensorView by value (trivially copyable).
    //
    //   kernel<<<g,b>>>(tensor.view());
    // ------------------------------------------------------------------
    __host__ __device__ TensorView<T, dim> view() {
        return static_cast<Base&>(*this);
    }

    __host__ __device__ TensorView<const T, dim> view() const {
        // Construct a const view manually since Base holds non-const T*.
        TensorView<const T, dim> v;
        v.data_ = this->data_;
        for (int i = 0; i < dim; ++i) {
            v.shape_[i]   = this->shape_[i];
            v.strides_[i] = this->strides_[i];
        }
        return v;
    }

    // ------------------------------------------------------------------
    // Owning-only utilities (not on TensorView)
    // ------------------------------------------------------------------

    void fill(T value) {
        const size_t n = this->numel();
        for (size_t i = 0; i < n; ++i) this->data_[i] = value;
    }

    void prefetch(int device = 0) {
        CUDA_CHECK(cudaMemPrefetchAsync(
            this->data_, this->numel() * sizeof(T), device));
    }

    static void sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

private:
    void compute_strides() {
        this->strides_[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i)
            this->strides_[i] = this->strides_[i + 1] * this->shape_[i + 1];
    }
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------
template <typename T> using Tensor1D = Tensor<T, 1>;
template <typename T> using Tensor2D = Tensor<T, 2>;
template <typename T> using Tensor3D = Tensor<T, 3>;

template <typename T> using TensorView1D = TensorView<T, 1>;
template <typename T> using TensorView2D = TensorView<T, 2>;
template <typename T> using TensorView3D = TensorView<T, 3>;