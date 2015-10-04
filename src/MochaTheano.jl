module MochaTheano
import Mocha: Config
export TheanoLossLayer
include("theano-layer.jl")

if Config.use_cuda
  include("cuda/theano-layer.jl")
end
end
