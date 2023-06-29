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
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            var act = GetEpsilonAct(Environ.Observation!.Value!);
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var step = Environ.Step(act, epoch);
                act = Update(step);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }
    }
}