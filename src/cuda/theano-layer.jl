using PyCall
@pyimport theano.sandbox.cuda as cuda
const long = pybuiltin("long")
function cudandarray(blob::CuTensorBlob)
  # Create a Theano CudaNdArray wrapper for the GPU blob data
  if eltype(blob) == Float64
    error("Only Float32 is supported for Theano GPU code")
  end
  ptr = long(blob.ptr.p)
  strides = [1 for s in blob.shape]
  return cuda.from_gpu_pointer(ptr, blob.shape, strides, blob)
end

function forward(backend::GPUBackend, state::TheanoLossLayerState, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  n = length(inputs[1])
  # map Mocha input to Theano - for CPUBlob, just get the array
  in = map(cudandarray, inputs)

  # apply the forward function to get the loss (it's returned as a 0-d array)
  # TODO how to extract the output from the GPU?

  loss = state.forward(in...)[]

  state.loss += loss*state.layer.weight

  # accumulate statistics
  state.loss_accum *= state.n_accum
  state.loss_accum += state.loss * n
  state.loss_accum /= state.n_accum + n

  state.n_accum += n
end

function backward(backend::GPUBackend, state::TheanoLossLayerState,
                  inputs::Vector{Blob}, diffs::Vector{Blob})
  # map Mocha input to Theano - for CPUBlob, just get the array
  in = map(cudandarray, inputs)

  for (g, diff) in zip(state.gradients, diffs)
    if isa(diff, CPUBlob)
      # TODO Get theano to output to diffs directly without copying

      copy!(diff, g(in...))
    end
  end
end
