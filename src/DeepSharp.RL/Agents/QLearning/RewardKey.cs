using DeepSharp.RL.Environs;
using DeepSharp.Utility.Operations;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     奖励表 复合键
    /// </summary>
    public struct RewardKey
    {
        public RewardKey(Observation state, Act act, Observation newState)
        {
            State = state;
            Act = act;
            NewState = newState;
        }

        public Observation State { set; get; }
        public Act Act { set; get; }
        public Observation NewState { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State.Value);
            var action = OpTensor.ToLongArrString(Act.Value);
            var newState = OpTensor.ToArrString(NewState.Value);

            return $"{state} \t {action} \t {newState}";
        }
    }
}