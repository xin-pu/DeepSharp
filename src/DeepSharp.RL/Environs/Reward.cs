namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     Reward
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
        ///     reward
        /// </summary>
        public float Value
        {
            set => SetProperty(ref _value, value);
            get => _value;
        }

        /// <summary>
        ///     TimeStamp of get reward
        /// </summary>
        public DateTime TimeStamp
        {
            set => SetProperty(ref _timeStamp, value);
            get => _timeStamp;
        }

        public override string ToString()
        {
            return $"Reward:\t{Value}";
        }
    }
}