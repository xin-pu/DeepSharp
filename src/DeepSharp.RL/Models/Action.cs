using DeepSharp.Utility.Operations;

namespace DeepSharp.RL.Models
{
    public class Action : ObservableObject, IEqualityComparer<Action>
    {
        private DateTime _timeStamp;
        private torch.Tensor? _value;

        public Action(torch.Tensor? action)
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

        public Action To(torch.Device device)
        {
            return new Action(Value?.to(device));
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{OpTensor.ToLongArrString(Value!)}";
        }

        public bool Equals(Action x, Action y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (ReferenceEquals(x, null)) return false;
            if (ReferenceEquals(y, null)) return false;
            if (x.GetType() != y.GetType()) return false;
            return x._timeStamp.Equals(y._timeStamp) && Equals(x._value, y._value);
        }

        public int GetHashCode(Action obj)
        {
            return HashCode.Combine(obj._timeStamp, obj._value);
        }
    }
}