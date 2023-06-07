using DeepSharp.RL.Models;
using DeepSharp.Utility.Operations;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     奖励表 复合键
    /// </summary>
    public struct RewardKey
    {
        public RewardKey(Observation state, Action action, Observation newState)
        {
            State = state;
            Action = action;
            NewState = newState;
        }

        public Observation State { set; get; }
        public Action Action { set; get; }
        public Observation NewState { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State.Value);
            var action = OpTensor.ToLongArrString(Action.Value);
            var newState = OpTensor.ToArrString(NewState.Value);

            return $"{state} \t {action} \t {newState}";
        }
    }
}