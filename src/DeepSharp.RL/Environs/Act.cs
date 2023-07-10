namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     动作
    /// </summary>
    public class Act : IEqualityComparer<Act>
    {
        public Act(torch.Tensor? action)
        {
            Value = action;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     奖励的张量格式
        /// </summary>
        public torch.Tensor? Value { set; get; }

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp { set; get; }


        public bool Equals(Act? x, Act? y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (ReferenceEquals(x, null)) return false;
            if (ReferenceEquals(y, null)) return false;
            return x.GetType() == y.GetType() && x.Value!.Equals(y.Value!);
        }

        public int GetHashCode(Act obj)
        {
            return HashCode.Combine(obj.TimeStamp, obj.Value);
        }

        public Act To(torch.Device device)
        {
            return new Act(Value!.to(device));
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{Value!.ToString(torch.numpy)}";
        }
    }
}