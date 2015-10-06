module Theano

export tensor, grad, TheanoVariable, TheanoFunction, theanofunction

import Base: (.*), (./), (.^), (+), (-), (*), (/), (^), log, exp, transpose, sum, log1p, mean, size, getindex, ndims

using PyCall
@pyimport theano
@pyimport theano.tensor as T

# TODO we should tighten this up with parameteric types for Theano variables (dimensions, eltype, maybe broadcastability)
typealias TheanoVariable{N} PyObject
typealias TheanoFunction    PyObject


(+)(x::TheanoVariable, y::TheanoVariable) = x[:__add__](y)
(-)(x::TheanoVariable, y::TheanoVariable) = x[:__sub__](y)
(.*)(x::TheanoVariable, y::TheanoVariable) = x[:__mul__](y)
(./)(x::TheanoVariable, y::TheanoVariable) = x[:__div__](y)
(.^){N<:Integer}(x::TheanoVariable, n::N)  = x[:__pow__](n)

(*){N<:Number}(x::TheanoVariable, y::N) = x[:__mul__](y)
(*){N<:Number}(x::N, y::TheanoVariable) = y[:__rmul__](x)

for fn in [:exp, :log, :log1p, :mean, :sum]
  @eval $fn(x::TheanoVariable) = T.$fn(x)
end

transpose(x::TheanoVariable) = ndims(x) = 1 ? T.shape_padleft(x) : T.transpose(x)

dot(x::TheanoVariable, y::TheanoVariable) = T.dot(x, y)
#tensordot(x::TheanoVariable, y::TheanoVariable, axes::Array{(Int,Int)}) = T.tensordot(x, y, axes)

sum(x::TheanoVariable, region::Int) = T.sum(x, region-1)
getindex(x::TheanoVariable, ix::Int) = x[:__getitem__](ix)
#getindex(x::TheanoVariable, ix::Tuple{}) = x[:__getitem__](())

grad(cost::TheanoVariable{1}, wrt::TheanoVariable...) = T.grad(cost, wrt)

theanofunction(inputs::Vector{TheanoVariable}, output::TheanoVariable) = theano.pymember(:function)(inputs, output, allow_input_downcast=true)


ndims(x :: TheanoVariable) = x[:ndim]

size(x::TheanoVariable) = T.shape(x)
# Follow the convention from Mocha.blobs that size(x, -1) gives size(x)[end], etc.
function size(x :: TheanoVariable, dim :: Int)
  N = ndims(x)
  if dim < 0
    dim = N+1 + dim
  end
  return T.shape(x)[dim-1]
end

DTYPES = [Float32 => "float32",
          Float64 => "float64"]

tensor(eltype::DataType, ndims::Int) = T.TensorType(DTYPES[eltype], repmat([false],ndims))()

end
