using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class SARSATest : AbstractTest
    {
        public SARSATest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void KArmedBanditMain()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75});
            var agent = new SARSA(kArmedBandit);
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
        public void FrozenlakeMain()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f});
            var agent = new SARSA(frozenlake);
            Print(frozenlake);


            var i = 0;
            float reward;
            const int testEpisode = 20;
            const float predReward = 0.7f;
            do
            {
                i++;
                frozenlake.Reset();
                agent.Learn();

                reward = agent.TestEpisodes(testEpisode);
            } while (reward < predReward);

            Print($"Stop after Learn {i}");
            frozenlake.ChangeToRough();
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}