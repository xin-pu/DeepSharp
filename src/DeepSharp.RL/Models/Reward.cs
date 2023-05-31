namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     一批观察得到的汇总奖励
    /// </summary>
    public class Reward : ObservableObject
    {
        private DateTime _timeStamp;
        private float _value;

        public Reward(float value)
        {
            Value = value;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     奖励产生的时间戳
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }

        /// <summary>
        ///     奖励,定量
        /// </summary>
        public float Value
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