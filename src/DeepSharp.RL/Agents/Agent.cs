using DeepSharp.RL.Environs;
using MathNet.Numerics.Random;

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
        public Act GetEpsilonAct(torch.Tensor state, double epsilon = 0.1)
        {
            var d = new SystemRandomSource();
            var v = d.NextDouble();
            var act = v < epsilon
                ? GetSampleAct()
                : GetPolicyAct(state);
            return act;
        }


        public abstract void Update(Episode episode);

        public abstract float Learn(int count);


        /// <summary>
        ///     Get Episode by Agent
        ///     以策略为主，运行得到一个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode RunEpisode(
            PlayMode playMode = PlayMode.Agent,
            Action<Step>? stepUpdate = null)
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
                stepUpdate?.Invoke(step);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }

        /// <summary>
        ///     Get Episodes by Agent
        ///     以策略为主，运行得到多个完整片段
        /// </summary>
        /// <returns>奖励</returns>
        public virtual Episode[] RunEpisode(int count,
            PlayMode playMode = PlayMode.Agent,
            Action<Step>? stepUpdate = null)
        {
            var episodes = new List<Episode>();
            foreach (var _ in Enumerable.Repeat(1, count))
                episodes.Add(RunEpisode(playMode, stepUpdate));

            return episodes.ToArray();
        }
    }

    public enum PlayMode
    {
        Sample,
        Agent,
        EpsilonGreedy
    }
}