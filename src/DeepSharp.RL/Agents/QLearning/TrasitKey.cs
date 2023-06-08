using DeepSharp.RL.Environs;
using DeepSharp.Utility.Operations;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     转移表 复合键
    /// </summary>
    public struct TrasitKey
    {
        public TrasitKey(Observation state, Action action)
        {
            State = state;
            Action = action;
        }

        public Observation State { set; get; }
        public Action Action { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State.Value);
            var action = OpTensor.ToLongArrString(Action.Value);

            return $"{state} \t {action}";
        }
    }
}