namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     观察
    /// </summary>
    public class Observation : ObservableObject
    {
        private DateTime _timeStamp;
        private torch.Tensor? _value;

        public Observation(torch.Tensor? state)
        {
            Value = state;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     观察的张量格式
        /// </summary>
        public torch.Tensor? Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        /// <summary>
        ///     观察产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }

        public Observation To(torch.Device device)
        {
            return new Observation(Value?.to(device));
        }

        public override string ToString()
        {
            if (Value is null)
                return string.Empty;
            var data = Value.data<float>().ToList();
            var dataStr = string.Join(",", data);
            return $"Observation:{dataStr}\t@{TimeStamp}";
        }
    }
}