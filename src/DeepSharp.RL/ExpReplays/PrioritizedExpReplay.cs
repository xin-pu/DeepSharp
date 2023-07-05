using DeepSharp.RL.Environs;

namespace DeepSharp.RL.ExpReplays
{
    /// <summary>
    ///     Uniform sample from Experience Source Cache
    /// </summary>
    public class PrioritizedExpReplay : ExpReplay
    {
        /// <summary>
        /// </summary>
        /// <param name="c">Capacity of Experience Replay Buffer,recommend 10^5 ~ 10^6</param>
        public PrioritizedExpReplay(int capacity = 10000)
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
            var probs = torch.from_array(Buffers.Select(a => a.Priority).ToArray());
            var randomIndex = torch.multinomial(probs, batchsize).data<long>().ToArray();

            var steps = randomIndex
                .AsParallel()
                .Select(i => Buffers.ElementAt((int) i))
                .ToArray();

            return steps;
        }
    }
}