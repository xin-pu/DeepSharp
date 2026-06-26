using RLSharp.FrozenLake.Web.Models;
using RLSharp.Torch.Environs;

namespace RLSharp.FrozenLake.Web.Services
{
	/// <summary>
	///     Extracts a GridState snapshot from RLSharp.Torch.Environs.FrozenLake + Step.
	/// </summary>
	public static class GridStateExtractor
	{
		public static GridState Extract(RLSharp.Torch.Environs.FrozenLake env, Step step)
		{
			var cells = new List<GridCell>();
			for (var i = 0; i < env.LakeUnits.Count; i++)
			{
				var unit = env[i];
				cells.Add(new GridCell
				{
					Index    = unit.Index,
					Row      = unit.Row,
					Column   = unit.Column,
					Role     = unit.Role.ToString(),
					IsPlayer = unit.Index == env.PlayID
				});
			}

			// Action tensor: 0=Up, 1=Down, 2=Left, 3=Right
			var actionIndex = step.Action?.Value is not null
				? step.Action.Value.ToInt32()
				: 0;

			return new GridState
			{
				Cells       = cells,
				PlayerIndex = env.PlayID,
				Action      = actionIndex,
				ActionName = actionIndex switch
				{
					0 => "Up",
					1 => "Down",
					2 => "Left",
					3 => "Right",
					_ => "?"
				},
				Reward     = step.Reward.Value,
				IsComplete = step.IsComplete
			};
		}
	}
}