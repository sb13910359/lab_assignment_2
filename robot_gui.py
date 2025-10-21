import numpy as np
import swift

class RobotGUI:
    """
    Multi-robot GUI controller for Swift environment
    Includes:
    - Manual vs Auto mode switching
    - Robot selection
    - Shared joint sliders
    """

    def __init__(self, env, robot1, robot2, robot3,
                 gripper_stick_arm, robot1_stick_base,
                 gripper_stick_arm2, update_robot3_ee,
                 state_dict,
                 set_estop_func=None, clear_estop_func=None):

        # Link callbacks for E-STOP
        self.set_estop_func = set_estop_func
        self.clear_estop_func = clear_estop_func

        self.env = env
        self.robot1 = robot1
        self.robot2 = robot2
        self.robot3 = robot3
        self.update_robot3_ee = update_robot3_ee
        self.gripper_stick_arm = gripper_stick_arm
        self.robot1_stick_base = robot1_stick_base
        self.gripper_stick_arm2 = gripper_stick_arm2
        self.state = state_dict

        # Default active robot
        self.active_robot = self.robot1

        # Build GUI elements 
        self._build_estop_buttons()
        self._build_mode_buttons()
        self._build_robot_select_buttons()
        self._build_sliders()
                     
        print("Press 🟡 Manual Mode to pause automation, then use Control Robot 1/2/3 + sliders to move each robot.")


    #  E-STOP SYSTEM  (per-robot)
    def _build_estop_buttons(self):

        def make_estop_button(robot_id):
            """Create per-robot E-STOP button with 3-state cycle."""

            states = [
                (f"E-STOP (Robot {robot_id})"),     # normal
                (f"🚨 E-STOP ACTIVE (Robot {robot_id})"),  # active
                (f"⚪ RELEASE E-STOP? (Robot {robot_id})") # confirm release
            ]
            current_state = {"idx": 0}

            def toggle(_=None):
                idx = current_state["idx"]

                # --- transition behaviour ---
                if idx == 0:        # going from normal to active
                    if self.set_estop_func:
                        self.set_estop_func(robot_id, True)
                    self.state[f"r{robot_id}_estop"] = True
                    print(f"🚨 E-STOP ENGAGED — Robot {robot_id} halted.")
                    current_state["idx"] = 1

                elif idx == 1:      # active to confirm release
                    current_state["idx"] = 2
                    print(f"⚪ Confirm release for Robot {robot_id}?")

                elif idx == 2:      # confirm to released
                    if self.clear_estop_func:
                        self.clear_estop_func(robot_id)
                    self.state[f"r{robot_id}_estop"] = False
                    print(f"✅ E-STOP cleared — Robot {robot_id} ready.")
                    current_state["idx"] = 0

                # update button label
                btn.desc = states[current_state["idx"]]

            # create button
            btn = swift.Button(desc=states[current_state["idx"]], cb=toggle)
            self.env.add(btn)
            return btn

        # create 3 E-STOP buttons
        self.estop_btn_r1 = make_estop_button(1)
        self.estop_btn_r2 = make_estop_button(2)
        self.estop_btn_r3 = make_estop_button(3)


    # MODE BUTTON

    def _build_mode_buttons(self):
        """Create one toggle button that switches between Manual and Auto mode."""

        # Initialize in current state (read from shared dict)
        initial_mode = self.state.get("mode", "auto")

        if initial_mode == "manual":
            desc = "🟡 Manual Mode"
        else:
            desc = "🟢 Auto Mode"

        def toggle_mode(_=None):
            """Switch between Manual and Auto modes."""
            if self.state["mode"] == "auto":
                # switch to manual
                self.state["mode"] = "manual"
                self.state["auto"] = False
                self.state["r1_patrol"] = False
                self.state["r2_active"] = False
                self.state["r3_active"] = False
                self.state["pick_and_place"] = False
                print("🟡 Switched to MANUAL MODE — sliders enabled for direct control.")
                mode_btn.desc = "🟡 Manual Mode"

            else:
                # switch to auto
                self.state["mode"] = "auto"
                self.state["auto"] = True
                self.state["r1_patrol"] = True
                self.state["r2_active"] = True
                self.state["r3_active"] = True
                print("🟢 Switched to AUTO MODE — autonomous sequences active.")
                mode_btn.desc = "🟢 Auto Mode"

        # Create the single toggle button
        mode_btn = swift.Button(desc=desc, cb=toggle_mode)
        self.env.add(mode_btn)

        # Store for reference 
        self.mode_btn = mode_btn



    # ROBOT SELECTION BUTTONS 
    def _build_robot_select_buttons(self):
        """Single button that cycles through Robot 1–3 each time it's clicked."""

        robots = [
            (self.robot1, "Robot 1 (Gen3 Lite)"),
            (self.robot2, "Robot 2 (UR3)"),
            (self.robot3, "Robot 3 (IRB1200)")
        ]
        self.robot_cycle_index = 0
        self.active_robot, name = robots[self.robot_cycle_index]

        # Create one button
        self.selected_robot_display = swift.Button(
            desc=f"Selected: {name}",
            cb=lambda _: cycle_robot()    # click cycles robot
        )
        self.env.add(self.selected_robot_display)

        def cycle_robot():
            """Cycle through robots when button is clicked."""
            self.robot_cycle_index = (self.robot_cycle_index + 1) % len(robots)
            self.active_robot, name = robots[self.robot_cycle_index]
            self.selected_robot_display.desc = f"Selected: {name}"
            print(f"Now controlling: {name}")

    def get_active_robot_id(self):
        """Return 1, 2, or 3 depending on which robot is currently selected."""
        if self.active_robot == self.robot1:
            return 1
        elif self.active_robot == self.robot2:
            return 2
        elif self.active_robot == self.robot3:
            return 3
        return None


    # SLIDERS
    def _build_sliders(self):
        def slider_callback(value_deg, joint_index):
            if self.state["mode"] != "manual" or self.state["e_stop"]:
                return
            
            q_new = np.deg2rad(float(value_deg))

            #check joint limits
            qmin, qmax = self.active_robot.qlim[0, joint_index], self.active_robot.qlim[1, joint_index]
            if q_new < qmin:
                q_new = qmin
            elif q_new > qmax:
                q_new = qmax

            q = self.active_robot.q.copy()
            q[joint_index] = q_new
            self.active_robot.q = q
            
            # update grippers / ee
            if self.active_robot == self.robot1:
                self.gripper_stick_arm()
                self.robot1_stick_base()
            elif self.active_robot == self.robot2:
                self.gripper_stick_arm2()
            elif self.active_robot == self.robot3:
                self.update_robot3_ee()

            self.env.step(0.02)

        for i in range(6):
            s = swift.Slider(
                cb=lambda v, j=i: slider_callback(v, j),
                min=-180, max=180, step=1,
                value=0,
                desc=f"Joint {i+1}",
                unit="°"
            )
            self.env.add(s)
            


