
from absl import app, flags
import shutil

FLAGS = flags.FLAGS
flags.DEFINE_integer('nx', 32, 'Number of pads in x direction')
flags.DEFINE_integer('ny', 32, 'Number of pads in y direction')


def main(_):
        
    dx = FLAGS.nx
    dy = FLAGS.ny

    size_x = 0.011/dx
    size_y = 0.009375*2/dy
            
    f = open("tactile_envs/assets/insertion/right_pad_collisions.xml", "w")
    f.write("<mujoco>\n")
    for i in range(dy):
        pos_y = size_y + 2*size_y*i
        for j in range(dx):

            pos_x = -0.011 + size_x + 2*size_x*j
            
            rgb = 0.6 + 0.1 * (i*dx + j)/(dx*dy - 1)

            xml_string = "<geom class=\"pad\" pos=\"{} -0.0026 {}\" size=\"{} 0.004 {}\" rgba=\"{} {} {} 1\"/>".format(pos_x, pos_y, size_x, size_y, rgb, rgb, rgb)
            
            f.write(xml_string + '\n')
    f.write("</mujoco>")
    f.close()
    shutil.copyfile("tactile_envs/assets/insertion/right_pad_collisions.xml", "tactile_envs/assets/insertion/left_pad_collisions.xml")

    touch_sensor_string = """
    <mujoco>
    <sensor>
        <plugin name="touch_right" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_right">
        <config key="size" value="{} {}"/>
            <config key="fov" value="14 23"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
    <sensor>
        <plugin name="touch_left" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_left">
        <config key="size" value="{} {}"/>
            <config key="fov" value="14 23"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="3"/>
        </plugin>
    </sensor>
    </mujoco>
    """.format(dx, dy, dx, dy)
    f = open("tactile_envs/assets/insertion/touch_sensors.xml", "w")
    f.write(touch_sensor_string)
    f.close()


if __name__ == "__main__":
    app.run(main)