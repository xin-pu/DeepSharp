using DeepSharp.RL.Models;
using static TorchSharp.torch.optim;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     An Agent base on CrossEntropy Function
    ///     Cross-Entropy Method
    ///     http://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf
    /// </summary>
    public class AgentCrossEntropyExt : AgentCrossEntropy
    {
        public AgentCrossEntropyExt(Environ environ, float percentElite = 0.7f, int hiddenSize = 100)
            : base(environ, percentElite, hiddenSize)
        {
            Optimizer = Adam(AgentNet.parameters(), 0.001);
        }


        /// <summary>
        ///     智能体 根据观察 生成动作 概率 分布，并按分布生成下一个动作
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Action PredictAction(Observation observation)
        {
            var input = observation.Value.unsqueeze(0);
            var sm = Softmax(1);
            var actionProbs = sm.forward(AgentNet.forward(input));
            var nextAction = torch.multinomial(actionProbs, SampleActionSpace);
            return new Action(nextAction);
        }

        /// <summary>
        ///     增加记忆功能，记录历史的精英片段
        /// </summary>
        public List<Episode> MemeSteps { set; get; } = new();

        public override float Learn(Episode[] steps)
        {
            var final = steps.Concat(MemeSteps)
                .OrderByDescending(a => a.SumReward.Value)
                .ToArray();

            var d = final.OrderByDescending(a => a.DateTime).Take(10);
            MemeSteps = d.OrderByDescending(a => a.SumReward.Value).Take(10).ToList();

            return base.Learn(final);
        }
    }
}