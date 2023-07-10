namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     Reward
    /// </summary>
    public class Reward
    {
        public Reward(float value)
        {
            Value = value;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     reward
        /// </summary>
        public float Value { set; get; }

        /// <summary>
        ///     TimeStamp of get reward
        /// </summary>
        public DateTime TimeStamp { set; get; }

        public override string ToString()
        {
            return $"Reward:\t{Value}";
        }
    }
}