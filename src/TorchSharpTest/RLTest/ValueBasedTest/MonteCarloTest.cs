using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class MonteCarloTest : AbstractTest
    {
        public MonteCarloTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void KABOnPolicyTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75});
            var agent = new MonteCarloOnPolicy(kArmedBandit);
            Print(kArmedBandit);

            var i = 0;
            float reward;
            const int testEpisode = 20;
            const float predReward = 17f;
            do
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();

                reward = agent.TestEpisodes(testEpisode);
            } while (reward < predReward);

            Print($"Stop after Learn {i}");

            var episode = agent.RunEpisode();
            Print(episode);
        }

        [Fact]
        public void KABOffPolicyTest()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75});
            var agent = new MonteCarloOffPolicy(kArmedBandit);
            Print(kArmedBandit);

            var i = 0;
            float reward;
            const int testEpisode = 20;
            const float predReward = 17f;
            do
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();

                reward = agent.TestEpisodes(testEpisode);
            } while (reward < predReward);

            Print($"Stop after Learn {i}");

            var episode = agent.RunEpisode();
            Print(episode);
        }

        [Fact]
        public void FLOnPolicyTest()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f});
            var agent = new MonteCarloOnPolicy(frozenlake, 0.1f, 50);
            Print(frozenlake);


            var i = 0;
            float reward = 0;
            const int testEpisode = 20;
            const float predReward = 0.7f;
            do
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                if (i % 100 == 0)
                {
                    reward = agent.TestEpisodes(testEpisode);
                    Print($"{i:D5}:\t{reward}");
                }
            } while (reward < predReward);

            Print($"Stop after Learn {i}");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
        }

        [Fact]
        public void FLOffPolictTest()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0f, 0f});
            var agent = new MonteCarloOffPolicy(frozenlake, 0.1f, 50);
            Print(frozenlake);


            var i = 0;
            float reward = 0;
            const int testEpisode = 20;
            const float predReward = 0.7f;
            do
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                if (i % 100 == 0)
                {
                    reward = agent.TestEpisodes(testEpisode);
                    Print($"{i:D5}:\t{reward}");
                }
            } while (reward < predReward);

            Print($"Stop after Learn {i}");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}