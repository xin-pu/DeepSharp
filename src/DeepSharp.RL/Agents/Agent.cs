using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     智能体
    /// </summary>
    public abstract class Agent : ObservableObject
    {
        protected Agent(Environ<Space, Space> env)
        {
            Environ = env;
            Device = env.Device;
        }

        public torch.Device Device { protected set; get; }

        public long ObservationSize => Environ.ObservationSpace!.N;

        public long ActionSize => Environ.ActionSpace!.N;

        public Environ<Space, Space> Environ { protected set; get; }

        /// <summary>
        ///     Select Action According with State
        ///     根据策略选择动作
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public abstract Act SelectAct(Observation state);

        public abstract void Update(Episode episode);

        public abstract float Learn(int count);


        /// <summary>
        ///     Get Episode by Agent
        ///     以策略为主，运行得到一个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode PlayEpisode(PlayMode playMode = PlayMode.Agent, bool updateAgent = false)
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var act = playMode switch
                {
                    PlayMode.Sample => Environ.SampleAct(),
                    PlayMode.Agent => SelectAct(Environ.Observation!),
                    _ => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
                };
                var step = Environ.Step(act, epoch);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward * Environ.DiscountReward(episode, Environ.Gamma);
            episode.SumReward = new Reward(sumReward);

            if (updateAgent) Update(episode);

            return episode;
        }

        /// <summary>
        ///     Get Episodes by Agent
        ///     以策略为主，运行得到多个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode[] PlayEpisode(int count, PlayMode playMode = PlayMode.Agent, bool updateAgent = false)
        {
            return Enumerable.Repeat(1, count).Select(a => PlayEpisode(playMode, updateAgent)).ToArray();
        }
    }

    public enum PlayMode
    {
        Sample,
        Agent,
        EpsilonGreedy
    }
}