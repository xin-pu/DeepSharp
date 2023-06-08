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
            State = state;
            Act = act;
        }

        public Observation State { set; get; }
        public Act Act { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State.Value);
            var action = OpTensor.ToLongArrString(Act.Value);

            return $"{state} \t {action}";
        }
    }
}