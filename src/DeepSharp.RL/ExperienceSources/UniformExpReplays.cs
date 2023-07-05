using DeepSharp.RL.Environs;

namespace DeepSharp.RL.ExperienceSources
{
    /// <summary>
    ///     Uniform sample from Experience Source Cache
    /// </summary>
    public class UniformExpReplays : ExpReplays
    {
        /// <summary>
        /// </summary>
        /// <param name="c">Capacity of Experience Replay Buffer,recommend 10^5 ~ 10^6</param>
        public UniformExpReplays(int capacity = 10000)
            : base(capacity)
        {
        }


        /// <summary>
        ///     Uniform sample batch size steps from Queue
        /// </summary>
        /// <param name="batchsize">batch size</param>
        /// <returns></returns>
        protected override Step[] SampleSteps(int batchsize)
        {
            var randomIndex = torch.randint(0, Size, new[] {batchsize}).data<long>().ToArray();

            var steps = randomIndex
                .Select(i => Buffers.ElementAt((int) i))
                .ToArray();

            return steps;
        }
    }
}