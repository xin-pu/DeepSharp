using DeepSharp.RL.CrossEntropy;
using DeepSharp.RL.Policies;
using static TorchSharp.torch.optim;

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

        public torch.Tensor PredictAction(torch.Tensor observation)
        {
            return Net.forward(observation);
        }


        public void Learn(List<torch.Tensor> observations, List<torch.Tensor> rewards)
        {
            if (observations.Count < 1000)
                return;

            var filterObservations = observations.Skip(observations.Count - 1000).Take(1000).ToList();
            var filterRewards = rewards.Skip(rewards.Count - 1000).Take(1000).ToList();
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