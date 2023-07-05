using DeepSharp.RL.Environs;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     An Agent base on CrossEntropy Function
    ///     Cross-Entropy Method
    ///     http://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf
    /// </summary>
    public class CrossEntropyExt : CrossEntropy

    {
        public int MemsEliteLength = 30;
        public List<DateTime> Start = new();

        public CrossEntropyExt(Environ<Space, Space> environ,
            int t,
            float percentElite = 0.7f,
            int hiddenSize = 100)
            : base(environ, t, percentElite, hiddenSize)
        {
            Optimizer = Adam(AgentNet.parameters(), 0.01);
        }

        /// <summary>
        ///     增加记忆功能，记录历史的精英片段
        /// </summary>
        internal List<Episode> MemeSteps { set; get; } = new();

        /// <summary>
        ///     Get Elite
        /// </summary>
        /// <param name="episodes"></param>
        /// <param name="percent"></param>
        /// <returns></returns>
        public override Episode[] GetElite(Episode[] episodes)
        {
            var current = episodes.Select(a => a.DateTime).ToList();
            Start.Add(current.Min());
            if (Start.Count >= 10)
                Start.RemoveAt(0);

            var combine = episodes.Concat(MemeSteps).ToList();
            var reward = combine
                .Select(a => a.SumReward.Value)
                .ToArray();
            var rewardP = reward.OrderByDescending(a => a)
                .Take((int) (reward.Length * PercentElite))
                .Min();

            var filterEpisodes = combine
                .Where(e => e.SumReward.Value > rewardP)
                .ToArray();

            MemeSteps = filterEpisodes.Where(a => a.DateTime > Start.Min())
                .OrderByDescending(a => a.SumReward.Value)
                .Take(MemsEliteLength)
                .ToList();

            return filterEpisodes;
        }
    }
}