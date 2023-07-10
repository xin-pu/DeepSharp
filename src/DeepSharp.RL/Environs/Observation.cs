namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     观察
    /// </summary>
    public class Observation
    {
        public Observation(torch.Tensor? state)
        {
            Value = state;
            TimeStamp = DateTime.Now;
        }

        /// <summary>
        ///     观察的张量格式
        /// </summary>
        public torch.Tensor? Value { set; get; }

        /// <summary>
        ///     观察产生的时间戳
        /// </summary>
        public DateTime TimeStamp { set; get; }

        public Observation To(torch.Device device)
        {
            return new Observation(Value?.to(device));
        }


        public object Clone()
        {
            return new Observation(Value) {TimeStamp = TimeStamp};
        }

        public override string ToString()
        {
            return $"Observation\r\n{Value?.ToString(torch.numpy)}";
        }
    }
}