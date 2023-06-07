using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using DeepSharp.Utility.Operations;

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
            var agent = new AgentCrossEntropy(kArmedBandit);

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
            var kArmedBandit = new KArmedBandit(2, device);

            kArmedBandit[0].Prob = 1;
            kArmedBandit[1].Prob = 1;
            Print(kArmedBandit);

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new AgentQLearning(kArmedBandit);


            agent.RunRandom(kArmedBandit, 200);
            Print(agent.Rewards.Keys.Count);
            Print(string.Join("\r\n", agent.Rewards.Keys));
            Print(agent.Transits
                .Sum(a => a.Value.Sum(b => b.Value)));

            foreach (var transitsKey in agent.Transits.Keys)
            {
                var dict = agent.Transits[transitsKey];
                foreach (var sonkey in dict.Keys)
                    Print($"{transitsKey}\t {OpTensor.ToArrString(sonkey.Value)} \t {dict[sonkey]}");
            }
        }
    }
}