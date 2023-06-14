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


        public abstract Act PredictAction(Observation state);

        /// <summary>
        ///     以策略为主，运行得到一个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode PlayEpisode()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var action = PredictAction(Environ.Observation!).To(Device);
                var step = Environ.Step(action, epoch);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.Observation; /// It's import for Update Observation
            }


            var sumReward = episode.Steps.Sum(a => a.Reward.Value) * Environ.DiscountReward(episode, Environ.Gamma);
            episode.SumReward = new Reward(sumReward);
            return episode;
        }

        public abstract float Learn(Episode[] steps);
    }
}