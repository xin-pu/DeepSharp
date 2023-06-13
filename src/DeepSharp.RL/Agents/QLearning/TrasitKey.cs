using DeepSharp.RL.Environs;
using DeepSharp.Utility.Operations;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     转移表 复合键
    /// </summary>
    public struct TrasitKey
    {
        public TrasitKey(Observation state, Act act)
        {
            State = state.Value!;
            Act = act.Value!;
        }

        public TrasitKey(torch.Tensor state, torch.Tensor act)
        {
            State = state;
            Act = act;
        }

        public torch.Tensor State { set; get; }
        public torch.Tensor Act { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State);
            var action = OpTensor.ToLongArrString(Act);

            return $"{state} \t {action}";
        }
    }
}