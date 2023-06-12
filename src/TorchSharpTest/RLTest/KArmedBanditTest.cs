using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest
{
    public class KArmedBanditTest : AbstractTest
    {
        public KArmedBanditTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void BanditCreateTest()
        {
            var count = 0;
            var bandit = new Bandit("A");
            var range = 100;
            foreach (var i in Enumerable.Range(0, range))
            {
                var res = bandit.Step();
                if (res > 0) count++;
                Print($"{i:D4}:{res}");
            }

            Print(count * 1f / range);
        }

        [Fact]
        public void KArmedBanditCreateTest()
        {
            var kArmedBandit = new KArmedBandit(5, new torch.Device(DeviceType.CUDA));
            Print(kArmedBandit);
        }

        [Fact]
        public void RandomPickup()
        {
            var probs = new[] {0.5f, 0.5f};

            foreach (var _ in Enumerable.Repeat(0, 10))
            {
                var res = torch.multinomial(torch.from_array(probs), 1);
                var index = res.item<long>();
                Print(index);
            }
        }

        [Fact]
        public void Main()
        {
            var epoch = 100;
            var episodesEachBatch = 20;
            var device = new torch.Device(DeviceType.CUDA);

            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(2, device);
            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentCrossEntropy<Space, Space>(kArmedBandit);

            /// Step 3 Learn and Optimize
            foreach (var i in Enumerable.Range(0, epoch))
            {
                var batch = kArmedBandit.GetMultiEpisodes(agent, episodesEachBatch);
                var eliteOars = agent.GetElite(batch); /// Get eliteOars 

                /// Agent Learn by elite observation & action
                var loss = agent.Learn(eliteOars);
                var rewardMean = batch.Select(a => a.SumReward.Value).Average();

                Print($"Epoch:{i:D4}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
            }
        }


        [Fact]
        public void QLearningRunRandom()
        {
            var device = new torch.Device(DeviceType.CUDA);

            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(2, device)
            {
                [0] = {Prob = 0.8},
                [1] = {Prob = 0.8}
            };

            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentQLearning<Space, Space>(kArmedBandit);
            agent.RunRandom(kArmedBandit, 500);
            agent.ValueIteration();

            Print(agent.Values.Select(a => $"{a.Key}\t{a.Value}").ToArray());
        }

        [Fact]
        public void QLearningMain()
        {
            var device = new torch.Device(DeviceType.CUDA);
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(2, device)
            {
                [0] = {Prob = 0.5},
                [1] = {Prob = 0.8}
            };
            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentQLearning<Space, Space>(kArmedBandit);
            Print(kArmedBandit);

            var i = 0;
            var bestReward = 0f;
            while (true)
            {
                agent.RunRandom(kArmedBandit, 100);
                agent.ValueIteration();

                var episode = kArmedBandit.GetEpisode(agent);
                var sum = episode.SumReward;
                bestReward = new[] {bestReward, sum.Value}.Max();
                Print($"{i++}\t reward:{sum.Value}");
                if (sum.Value > 18)
                    break;
            }
        }
    }
}