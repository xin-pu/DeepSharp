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
            Net = new Net(obsSize, 10, actionSize);
            optimizer = Adam(Net.parameters());
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
            var action = Net.forward(observation.Value);
            var actionProbs = sm.forward(action);

            var nextAction = torch.multinomial(actionProbs, 1, false);

            return new Action {Value = nextAction};
        }


        /// <summary>
        /// </summary>
        /// <param name="observations">网络的输入 是单个观察</param>
        /// <param name="actions">网络的输出 是动作的概率分布</param>
        public void Learn(List<torch.Tensor> observations, List<torch.Tensor> actions)
        {
            if (observations.Count < 100)
                return;

            var filterObservations = observations.Skip(observations.Count - 1000).Take(1000).ToList();
            var filterRewards = actions.Skip(actions.Count - 1000).Take(1000).ToList();
            var observatio = torch.vstack(filterObservations);
            var reward = torch.vstack(filterRewards);
            var pred = Net.forward(reward);

            var output = Loss.call(pred, observatio);

            optimizer.zero_grad();
            output.backward();
            optimizer.step();

            var loss = output.item<float>();
        }
    }
}