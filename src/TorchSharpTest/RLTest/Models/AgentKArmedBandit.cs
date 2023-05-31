using DeepSharp.RL.CrossEntropy;
using DeepSharp.RL.Models;
using DeepSharp.RL.Policies;
using static TorchSharp.torch.optim;
using Action = DeepSharp.RL.Models.Action;

namespace TorchSharpTest.RLTest
{
    public class AgentKArmedBandit : IPolicy
    {
        public AgentKArmedBandit(int obsSize, int actionSize)
        {
            Net = new Net(obsSize, 100, actionSize);
            optimizer = Adam(Net.parameters(), 1E-2);
            Loss = CrossEntropyLoss();
        }

        public Net Net { set; get; }

        public Optimizer optimizer { set; get; }

        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { set; get; }


        /// <summary>
        ///     智能体 根据观察 生成动作 概率 分布，并按分布生成下一个动作
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public Action PredictAction(Observation observation)
        {
            var sm = Softmax(1);
            var action = Net.forward(observation.Value.unsqueeze(0));
            var actionProbs = sm.forward(action);

            var nextAction = torch.multinomial(actionProbs, 1, false);

            return new Action {Value = nextAction};
        }


        public OAR[] GetElite(Episode[] episodes, double percent)
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

            optimizer.zero_grad();
            output.backward();
            optimizer.step();

            var loss = output.item<float>();
            return loss;
        }
    }
}