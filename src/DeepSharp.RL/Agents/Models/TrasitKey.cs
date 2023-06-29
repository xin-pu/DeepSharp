using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Key of Transit, which combine state and action.
    /// </summary>
    public class TransitKey
    {
        public TransitKey(Observation state, Act act)
        {
            State = state.Value!;
            Act = act.Value!;
        }

        public TransitKey(torch.Tensor state, torch.Tensor act)
        {
            State = state;
            Act = act;
        }

        public torch.Tensor State { protected set; get; }
        public torch.Tensor Act { protected set; get; }


        public static bool operator ==(TransitKey x, TransitKey y)
        {
            return x.Equals(y);
        }

        public static bool operator !=(TransitKey x, TransitKey y)
        {
            return !x.Equals(y);
        }

        public bool Equals(TransitKey other)
        {
            return State.Equals(other.State) && Act.Equals(other.Act);
        }

        public override bool Equals(object? obj)
        {
            if (obj is TransitKey input)
                return Equals(input);
            return false;
        }

        public override int GetHashCode()
        {
            return -1;
        }

        public override string ToString()
        {
            return $"{State.ToString(torch.numpy)},{Act.ToString(torch.numpy)}";
        }
    }
}