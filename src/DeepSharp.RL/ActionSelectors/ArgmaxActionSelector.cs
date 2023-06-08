using FluentAssertions;

namespace DeepSharp.RL.ActionSelectors
{
    /// <summary>
    ///     传入张量，对最后一维执行 Argmax
    /// </summary>
    public class ArgmaxActionSelector : ActionSelector
    {
        /// <summary>
        /// </summary>
        /// <param name="probs"></param>
        /// <returns>action of long format</returns>
        public override torch.Tensor Select(torch.Tensor probs)
        {
            probs.dim().Should().Be(2, "ArgmaxActionSelector Support tensor which dims is 2");
            return torch.argmax(probs, -1, KeepDims);
        }
    }
}