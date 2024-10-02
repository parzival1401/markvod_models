using MarkovModels
using Plots
using SparseArrays
using Revise



T = Float32
SF = LogSemifield{T} 


fsm = VectorFSM{SF}()
S = 3 # Number of states

labels = ["a", "b", "c"]

prev = addstate!(fsm, labels[1], initweight = one(SF))
addarc!(fsm, prev, prev)
for s in 2:S
    fw = s == S ? one(SF) :  zero(SF)
    state = addstate!(fsm, labels[s]; finalweight = fw)
    addarc!(fsm, prev, state)
    addarc!(fsm, state, state)
    prev = state
end

fsm |> renormalize