using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Advanced techniques of Value Iterate
    /// </summary>
    public class ExperienceReplayBuffer
    {
        /// <summary>
        /// </summary>
        /// <param name="c">Capacity of Experience Replay Buffer,recommend 10^5 ~ 10^6</param>
        public ExperienceReplayBuffer(int c = 10000)
        {
            Capacity = c;
            Buffers = new Queue<Step>(c);
        }

        /// <summary>
        ///     Capacity of Experience Replay Buffer
        /// </summary>
        public int Capacity { protected set; get; }

        /// <summary>
        ///     Cache
        /// </summary>
        public Queue<Step> Buffers { set; get; }

        /// <summary>
        ///     Record a step [State, Action, Reward, NewState]
        /// </summary>
        /// <param name="step"></param>
        public void Append(Step step)
        {
            if (Buffers.Count == Capacity) Buffers.Dequeue();
            Buffers.Enqueue(step);
        }

        /// <summary>
        ///     Record steps {[State , Action, Reward, NewState],...,[State , Action, Reward, NewState]}
        /// </summary>
        /// <param name="steps"></param>
        public void Append(IEnumerable<Step> steps)
        {
            steps.ToList().ForEach(Append);
        }

        /// <summary>
        ///     Record steps by insert a episode
        /// </summary>
        /// <param name="episode"></param>
        public void Append(Episode episode)
        {
            Append(episode.Steps);
        }


        /// <summary>
        ///     sample n steps from Queue
        /// </summary>
        /// <param name="n">batch size</param>
        /// <returns></returns>
        public Step[] Sample(int n)
        {
            var length = Buffers.Count;
            var probs = torch.from_array(Enumerable.Repeat(1, length).ToArray(), torch.ScalarType.Float32);
            var randomIndex = torch.multinomial(probs, n).data<long>().ToArray();

            var steps = randomIndex
                .Select(i => Buffers.ElementAt((int) i))
                .ToArray();

            return steps;
        }
    }
}