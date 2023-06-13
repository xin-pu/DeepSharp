using DeepSharp.RL.Environs;
using DeepSharp.Utility.Operations;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     奖励表 复合键
    /// </summary>
    public struct RewardKey
    {
        public RewardKey(torch.Tensor state, torch.Tensor act, torch.Tensor newState)
        {
            State = state;
            Act = act;
            NewState = newState;
        }

        public RewardKey(Observation state, Act act, Observation newState)
        {
            State = state.Value!;
            Act = act.Value!;
            NewState = newState.Value!;
        }

        public torch.Tensor State { set; get; }
        public torch.Tensor Act { set; get; }
        public torch.Tensor NewState { set; get; }


        public override string ToString()
        {
            var state = OpTensor.ToArrString(State);
            var action = OpTensor.ToLongArrString(Act);
            var newState = OpTensor.ToArrString(NewState);

            return $"{state} \t {action} \t {newState}";
        }
    }
}