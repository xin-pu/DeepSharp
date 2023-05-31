using DeepSharp.RL.CrossEntropy;
using DeepSharp.RL.Policies;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Models.Agents
{
    public class AgentCrossEntropy : IPolicy
    {
        public AgentCrossEntropy(int obsSize, int actionSize, int hiddenSize = 100)
        {
            Net = new Net(obsSize, hiddenSize, actionSize);
            Optimizer = Adam(Net.parameters(), 1E-2);
            Loss = CrossEntropyLoss();
        }

        public Net Net { protected set; get; }

        public Optimizer Optimizer { protected set; get; }

        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { protected set; get; }

        public void UpdateOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
        }

        /// <summary>
        ///     智能体 根据观察 生成动作 概率 分布，并按分布生成下一个动作
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public Action PredictAction(Observation observation)
        {
            var input = observation.Value.unsqueeze(0);
            var sm = Softmax(1);
            var actionProbs = sm.forward(Net.forward(input));
            var nextAction = torch.multinomial(actionProbs, 1);
            return new Action(nextAction);
        }


        public Step[] GetElite(Episode[] episodes, double percent)
        {
            var reward = episodes
                .Select(a => a.SumReward.Value)
                .ToArray();
            var rewardP = reward.OrderByDescending(a => a)
                .Take((int) (reward.Length * percent))
                .Min();

            var elite = episodes
                .Where(e => e.SumReward.Value > rewardP)
                .SelectMany(e => e.Oars)
                .ToArray();

            return elite;
        }

        /// <summary>
        /// </summary>
        /// <param name="observations">网络的输入 是单个观察</param>
        /// <param name="actions">网络的输出 是动作的概率分布</param>
        public float Learn(torch.Tensor observations, torch.Tensor actions)
        {
            var pred = Net.forward(observations);
            var output = Loss.call(pred, actions);

            Optimizer.zero_grad();
            output.backward();
            Optimizer.step();

            var loss = output.item<float>();
            return loss;
        }
    }
}