using System.Diagnostics;

namespace RLSharp.Torch.ActionSelectors
{
	/// <summary>
	///     Performs argmax on the last dimension of the input tensor.
	/// </summary>
	public class ArgmaxActionSelector : ActionSelector
	{
		/// <summary>
		/// </summary>
		/// <param name="probs"></param>
		/// <returns>Action index in long format.</returns>
		public override torch.Tensor Select(torch.Tensor probs)
		{
			Debug.Assert(probs.dim() == 2, "ArgmaxActionSelector Support tensor which dims is 2");
			return torch.argmax(probs, -1, KeepDims);
		}
	}
}