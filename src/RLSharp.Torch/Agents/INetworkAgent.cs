namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Neural network Agent interface, providing optimizer access and checkpoint functionality.
	/// </summary>
	public interface INetworkAgent
	{
		Optimizer Optimizer { get; }

		void SaveCheckpoint(string dir);
		void LoadCheckpoint(string dir);
	}
}