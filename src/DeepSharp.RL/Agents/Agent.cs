using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;
using MathNet.Numerics.Random;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent
    {
        protected Agent(Environ<Space, Space> env, string name)
        {
            Name = name;
            Environ = env;
            Device = env.Device;
        }


        public string Name { protected set; get; }
        public torch.Device Device { protected set; get; }

        public long ObservationSize => Environ.ObservationSpace!.N;

        public long ActionSize => Environ.ActionSpace!.N;

        public Environ<Space, Space> Environ { protected set; get; }
        public float Epsilon { set; get; } = 0.2f;


        public abstract LearnOutcome Learn();

        public abstract void Save(string path);

        public abstract void Load(string path);


        /// <summary>
        ///     Get a random Action
        /// </summary>
        /// <returns></returns>
        public Act GetSampleAct()
        {
            return Environ.SampleAct();
        }

        /// <summary>
        ///     Get a action by Policy
        ///     π(s）
        /// </summary>
        /// <param name="state">current state</param>
        /// <returns>a action provide by agent's policy</returns>
        public abstract Act GetPolicyAct(torch.Tensor state);


        /// <summary>
        ///     Get a action by ε-greedy method
        ///     π^ε(s）
        /// </summary>
        /// <param name="state">current state</param>
        /// <param name="epsilon"></param>
        /// <returns></returns>
        public Act GetEpsilonAct(torch.Tensor state)
        {
            var d = new SystemRandomSource();
            var v = d.NextDouble();
            var act = v < Epsilon
                ? GetSampleAct()
                : GetPolicyAct(state);
            return act;
        }


        /// <summary>
        ///     Get Episode by Agent
        ///     以策略为主，运行得到一个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode RunEpisode(
            PlayMode playMode = PlayMode.Agent)
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var act = playMode switch
                {
                    PlayMode.Sample => GetSampleAct(),
                    PlayMode.Agent => GetPolicyAct(Environ.Observation!.Value!),
                    PlayMode.EpsilonGreedy => GetEpsilonAct(Environ.Observation!.Value!),
                    _ => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
                };
                var step = Environ.Step(act, epoch);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.PostState; /// It's import for Update Observation
            }

            var sumReward = Environ.GetReturn(episode);
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        /// <summary>
        ///     Get Episodes by Agent
        ///     以策略为主，运行得到多个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode[] RunEpisodes(int count,
            PlayMode playMode = PlayMode.Agent)
        {
            var episodes = new List<Episode>();
            foreach (var _ in Enumerable.Repeat(1, count))
                episodes.Add(RunEpisode(playMode));

            return episodes.ToArray();
        }

        /// <summary>
        /// </summary>
        /// <param name="testCount">test count</param>
        /// <returns>Average Reward</returns>
        public float TestEpisodes(int testCount)
        {
            var episode = RunEpisodes(testCount);
            var averageReward = episode.Average(a => a.SumReward.Value);
            return averageReward;
        }


        public override string ToString()
        {
            return $"Agent[{Name}]";
        }
    }
}