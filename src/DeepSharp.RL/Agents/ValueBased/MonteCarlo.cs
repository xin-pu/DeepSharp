using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     According with
    ///     [Neural Networks and Deep Learning](XiPeng Qiu)
    /// </summary>
    public class MonteCarlo : ValueAgent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="n">the leaning count of  each epoch </param>
        public MonteCarlo(Environ<Space, Space> env, int n = 10)
            : base(env, "MonteCarlo")
        {
            N = n;
        }

        public int N { protected set; get; }


        public Episode[] Learn()
        {
            var episodes = Enumerable.Range(0, N)
                .Select(i => RunEpisode())
                .ToList();

            // Todo Update Value Table Here 

            return episodes.ToArray();
        }
    }
}