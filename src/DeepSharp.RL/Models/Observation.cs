namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     观察
    /// </summary>
    public class Observation : ObservableObject
    {
        private DateTime _timeStamp;
        private torch.Tensor _value;

        /// <summary>
        ///     观察产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }


        /// <summary>
        ///     观察的张量格式
        /// </summary>
        public torch.Tensor Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        public override string ToString()
        {
            return $"{TimeStamp}\t{Value}";
        }
    }
}