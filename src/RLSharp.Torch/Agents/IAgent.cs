using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Core Agent interface defining the basic contract for all agents.
	/// </summary>
	public interface IAgent
	{
		string Name { get; }

		torch.Device Device { get; }

		EnvironmentBase<Space, Space> EnvironmentBase { get; }

		LearnOutcome Learn();
		void         Save(string               path);
		void         Load(string               path);
		ActionValue          GetPolicyAct(torch.Tensor state);
	}
}