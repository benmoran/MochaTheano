using Mocha
import Mocha: @defstruct, @characterize_layer, destroy, setup, shutdown, forward, backward, @info

using Theano
using PyCall

# TODO i think we can get rid of variables
# and just auto-construct theano variables of the same names?
# use a macro to define output

# HACK - We need to make TheanoVariable available in the Mocha package
# scope, in order to use the @Mocha.defstruct macro.
#Mocha.eval(:(using Theano))

@defstruct TheanoLossLayer Layer (
  name :: AbstractString = "TheanoLoss",
  (weight :: AbstractFloat = 1.0, weight >= 0),
  (bottoms :: Vector{Symbol} = Symbol[],
   length(bottoms) >= 1),
  eltype :: DataType = Float32,
  loss :: Expr = :(),
                                  )

@characterize_layer(TheanoLossLayer,
  can_do_bp => true,
  has_loss => true,
  is_sink   => true,
  has_stats => true,
)

type TheanoLossLayerState{T<:Number} <: LayerState
  layer      :: TheanoLossLayer
  loss       :: T
  loss_accum :: T
  n_accum    :: Int

  variablemod  :: Module
  forward    :: TheanoFunction
  gradients  :: Vector{TheanoFunction}
end

function reset_statistics(state::TheanoLossLayerState)
  state.n_accum = 0
  state.loss_accum = zero(typeof(state.loss_accum))
end

function dump_statistics(storage, state::TheanoLossLayerState, show::Bool)
  update_statistics(storage, "$(state.layer.name)-encoder-loss", state.loss_accum)

  if show
    loss = @sprintf("%.4f", state.loss_accum)
    @info("  $(state.layer.name)-loss (avg over $(state.n_accum)) = $loss")
  end
end

function setup(backend::Backend, layer::TheanoLossLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  data_type = eltype(inputs[1])

  # in the loss, replace all the input label names with theano variable references
  # create a variable of the right dimension and eltype for each input
  # Do we do this by module and eval?
  mod = Module()
  eval(mod, quote
    using Theano
    variables = TheanoVariable[]
  end)
  for (b, i) in zip(layer.bottoms, inputs)
    ex = quote
      $b = tensor($(layer.eltype), $(ndims(i)))
      push!(variables, $b)
    end
    eval(mod, ex)
  end
  loss = eval(mod, layer.loss)

  fwd = theanofunction(mod.variables, loss)
  grads = TheanoFunction[theanofunction(mod.variables, g) for g in grad(loss, mod.variables...)]
  state = TheanoLossLayerState(layer, zero(data_type), zero(data_type), 0, mod, fwd, grads)
  return state
end

function shutdown(backend::Backend, state::TheanoLossLayerState)
  # pass
  ## for blob in values(state.tmp_blobs)
  ##   destroy(blob)
## end
end

function forward(backend::CPUBackend, state::TheanoLossLayerState, inputs::Vector{Blob})
  data_type = eltype(inputs[1])
  n = length(inputs[1])
  # map Mocha input to Theano - for CPUBlob, just get the array
  in = [i.data for i in inputs]

 # apply the forward function to get the loss (it's returned as a 0-d array)
  loss = state.forward(in...)[]

  state.loss += loss*state.layer.weight

  # accumulate statistics
  state.loss_accum *= state.n_accum
  state.loss_accum += state.loss * n
  state.loss_accum /= state.n_accum + n

  state.n_accum += n
end

function backward(backend::CPUBackend, state::TheanoLossLayerState,
                  inputs::Vector{Blob}, diffs::Vector{Blob})
  # map Mocha input to Theano - for CPUBlob, just get the array
  in = [i.data for i in inputs]

  for (g, diff) in zip(state.gradients, diffs)
    if isa(diff, CPUBlob)
      # TODO Get theano to output to diffs directly without copying
      copy!(diff, g(in...))
    end
  end
end
