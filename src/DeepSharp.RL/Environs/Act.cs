using DeepSharp.Utility.Operations;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     动作
    /// </summary>
    public class Act : ObservableObject, IEqualityComparer<Act>
    {
        private DateTime _timeStamp;
        private torch.Tensor? _value;

        public Act(torch.Tensor? action)
        {
            Value = action;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     奖励的张量格式
        /// </summary>
        public torch.Tensor? Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }

        public bool Equals(Act? x, Act? y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (ReferenceEquals(x, null)) return false;
            if (ReferenceEquals(y, null)) return false;
            return x.GetType() == y.GetType() && x.Value!.Equals(y.Value!);
        }

        public int GetHashCode(Act obj)
        {
            return HashCode.Combine(obj._timeStamp, obj._value);
        }

        public Act To(torch.Device device)
        {
            return new Act(Value?.to(device));
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{OpTensor.ToLongArrString(Value!)}";
        }
    }
}