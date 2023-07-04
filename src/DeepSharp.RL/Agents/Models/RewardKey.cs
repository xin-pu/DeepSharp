using DeepSharp.RL.Environs;

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
            var state = State.ToString(torch.numpy);
            var action = Act.ToString(torch.numpy);
            var newState = NewState.ToString(torch.numpy);

            return $"{state} \r\n {action}  \r\n {newState}";
        }

        public static bool operator ==(RewardKey x, RewardKey y)
        {
            return x.Equals(y);
        }

        public static bool operator !=(RewardKey x, RewardKey y)
        {
            return !x.Equals(y);
        }

        public bool Equals(RewardKey other)
        {
            return State.Equals(other.State) && Act.Equals(other.Act) && NewState.Equals(other.NewState);
        }


        public override bool Equals(object? obj)
        {
            if (obj is RewardKey input)
                return Equals(input);
            return false;
        }

        public override int GetHashCode()
        {
            return -1;
        }
    }
}