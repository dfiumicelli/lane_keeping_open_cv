FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-image-transport \
    ros-humble-geometry-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

COPY src/turtlebot3_line_follower/package.xml ./src/turtlebot3_line_follower/

RUN apt-get update && \
    rosdep update && \
    rosdep install -r --from-paths src -i -y --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

COPY src ./src

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=0" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc

# Script entrypoint corretto
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
