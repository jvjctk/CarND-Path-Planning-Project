#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"


#include <map>
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

using std::map;

//********************* definitions**********************

// Weights of costs
const double W_DISTANCE = 50; 
const double W_SPEED    = 40;

const double SPEED_LIMIT = 22.3; // 50mph
const double BUFFER_V = 0.2;     // 49.5 mph (22.1 m/s)

const double BUFFER_DIST = 30;   // Safty distance in meter


const double SPEED_CHANGE = 0.1;       // accelation =  5 m/(s^2) 
const double MAX_SPEED_CHANGE = 0.19;  // accelation < 10 m/(s^2)
const double MAX_TARGET_SPEED = SPEED_LIMIT - BUFFER_V;  // 24.5-2.5 = 22 m/s

const double CLOSE_AHEAD  = 50;        // closest distance to check if need to change lane

const double MAX_COST = std::numeric_limits<const double>::infinity();

struct EgoVehicle
{
    
    double x;
    double y;
    double s;
    double d;
    double yaw;
    double speed;

};

struct VehicleNearby
{
    double dist;   // distance between the other car and ego vehicle
    double speed;  // speed of the vehicle
};

struct EgoParams
{
    int lane;
    double target_speed;
    bool ChangeLane;
};

struct MapWayPoints {
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;
};

struct PreviousPath {
  vector<double> previous_path_x;
  vector<double> previous_path_y;
};


map<string, VehicleNearby> Predict(const vector<vector<double>> &sensor_fusion, double car_s, int previous_path_size, int lane)
{

    double min_s = 9999;
    double max_s = -1; 

    double min_left_s  = 9999;
    double max_left_s  = -1;

    double min_right_s = 9999;   
    double max_right_s = -1;

    // vehicles that are near to ego vehicle
    VehicleNearby vehicle_ahead = {9999,0};
    VehicleNearby vehicle_behind = {9999,0};
    VehicleNearby vehicle_left_ahead   = {9999,0};
    VehicleNearby vehicle_left_behind  = {9999,0};
    VehicleNearby vehicle_right_ahead  = {9999,0};
    VehicleNearby vehicle_right_behind = {9999,0};
   
    // analyzing sensor fusion data
    for(int i = 0; i < sensor_fusion.size(); i++)
	{
        
        // position and speed of checked vehicle
        double vx = sensor_fusion[i][3];
        double vy = sensor_fusion[i][4];
        double check_speed = sqrt(vx*vx + vy*vy);
        double check_car_s = sensor_fusion[i][5];
        float d = sensor_fusion[i][6];

        // calculation of s of each checked vehicle
        check_car_s += ((double) previous_path_size * 0.02 * check_speed);
        // distance to my vehicle
        double dist = check_car_s - car_s;

        // if checked vehicle is on same lane
        if ( d > (2+4*lane-2) && d < (2+4*lane+2) ){              
            // if vehicle ahead and close
            if ( dist > 0 &&  check_car_s < min_s){
                vehicle_ahead.dist  = dist;
                vehicle_ahead.speed = check_speed;
                min_s = check_car_s;
            } 
            // car is behind and closed to my car
            else if (dist < 0 && check_car_s > max_s) { 
                vehicle_behind.dist  = abs(dist);
                vehicle_behind.speed = check_speed;
                max_s = check_car_s;
            }   
        }

 
        // if checked vehicle is on ego vehicle's left lane
        else if ( lane > 0 && d > (4*lane-4) && d < (4*lane) ) {
	    // if vehicle ahead and close
            if ( dist > 0 &&  check_car_s < min_left_s){
                vehicle_left_ahead.dist  = dist;
                vehicle_left_ahead.speed = check_speed;
                min_left_s = check_car_s;
            	} 
            // car is behind and closed to my car
            else if (dist < 0 && check_car_s > max_left_s) { 
                vehicle_left_behind.dist  = abs(dist);
                vehicle_left_behind.speed = check_speed;
                max_left_s = check_car_s;
            }   
        } 


        // if checked vehicle is on ego vehicle's right lane
        else {
            // if car is ahead and closest to my car
            if ( dist > 0 &&  check_car_s < min_right_s){
                vehicle_right_ahead.dist  = dist;
                vehicle_right_ahead.speed = check_speed;
                min_right_s = check_car_s;
            } 
            // car is behind and closed to my car
            else if (dist < 0 && check_car_s > max_right_s) { 
                vehicle_right_behind.dist  = abs(dist);
                vehicle_right_behind.speed = check_speed;
                max_right_s = check_car_s;
            }   
        } 
         
    }

    map<string, VehicleNearby> predictions;

    predictions["Ahead"]  = vehicle_ahead;
    predictions["Behind"] = vehicle_behind;
    predictions["Left_Ahead"]  = vehicle_left_ahead;
    predictions["Left_Behind"] = vehicle_left_behind;
    predictions["Right_Ahead"] = vehicle_right_ahead;
    predictions["Right_Behind"] = vehicle_right_behind;

    return predictions;
}

// distance cost calculation
double Calc_distance_cost(double dist) 
{
    double cost;
    
    if (dist < BUFFER_DIST) { // safety buffer
        cost = 35 - dist;
    } 
    else {
        cost = (1.0/dist);
    }
    return cost;

}

//speed cost calculation
double Calc_speed_cost(double ahead_speed, double my_speed) {
    double max_target_speed = SPEED_LIMIT - BUFFER_V;
    double cost;

    if ( ahead_speed >= my_speed ) {
        cost = (max_target_speed - my_speed) / max_target_speed;
    } else  {
        cost = my_speed - ahead_speed ;
    } 
    return cost;
}

// total cost calculation
double Calculate_cost(double dist_ahead, double dist_behind, double ahead_speed, double my_speed) 
{
    double cost_dist = std::max(Calc_distance_cost(dist_ahead) , Calc_distance_cost(dist_behind));
    double cost_speed = Calc_speed_cost(ahead_speed, my_speed);

    double total_cost = W_DISTANCE * cost_dist + W_SPEED * cost_speed;

    return total_cost;

}

double Update_target_speed(double ahead_dist, double ahead_speed, double target_speed)
{
     // if no vehicle ahead or far away
    if ( ahead_dist == 9999 || ahead_dist > CLOSE_AHEAD ) {
        if ( target_speed < MAX_TARGET_SPEED ) { // increase speed
            target_speed += SPEED_CHANGE;
        }
    }
    // faster than vehicle ahead, need to slow down
    else if ( target_speed > ahead_speed ){                   
        if ( (target_speed - ahead_speed) <= MAX_SPEED_CHANGE ){
            target_speed  = ahead_speed;
        } else {
            target_speed -= MAX_SPEED_CHANGE;
        }          
    }         
    return target_speed;
} 

EgoParams Plan(const map<string, VehicleNearby> &predictions, double target_speed, int lane) 
{
   
    // Retrieve predicted cars
    VehicleNearby vehicle_ahead  = predictions.find("Ahead")->second;
    VehicleNearby vehicle_behind = predictions.find("Behind")->second;
    VehicleNearby vehicle_left_ahead   = predictions.find("Left_Ahead")->second; 
    VehicleNearby vehicle_left_behind  = predictions.find("Left_Behind")->second;
    VehicleNearby vehicle_right_ahead  = predictions.find("Right_Ahead")->second;
    VehicleNearby vehicle_right_behind = predictions.find("Right_Behind")->second;

    double cost_keep, cost_left, cost_right;
    vector<double> total_costs = {};
    
    EgoParams egoparams = {lane, target_speed, false};

    // No vehicles nearby
    if ( vehicle_ahead.dist == 9999 ||  vehicle_ahead.dist > CLOSE_AHEAD ) {

       if ( target_speed < MAX_TARGET_SPEED ) { // able to speed up
           target_speed += SPEED_CHANGE;
       }
       egoparams.lane = lane;
       egoparams.target_speed = target_speed;
        
       return egoparams; 
    }

    // if vehicle is close, options to perform:  Keep Lane, Change to Left , Change to Right 

    // Calculate cost for Keep Lane  
    cost_keep = Calculate_cost(vehicle_ahead.dist, vehicle_behind.dist, vehicle_ahead.speed, target_speed);
    total_costs.push_back(cost_keep);    
  
    // Calculate cost for Change to Left lane
    if ( lane > 0 ) {  
        cost_left = Calculate_cost(vehicle_left_ahead.dist, vehicle_left_behind.dist, vehicle_left_ahead.speed, target_speed);      
    }
    else {
        cost_left = MAX_COST;      
    } 
    total_costs.push_back(cost_left);
   
    //Calculate cost for Change to right lane
    if ( lane < 2 ) {
        cost_right = Calculate_cost(vehicle_right_ahead.dist, vehicle_right_behind.dist, vehicle_right_ahead.speed, target_speed);      
    } else {
        cost_right = MAX_COST;
    }     
    total_costs.push_back(cost_right);


    // Find min cost
    vector<double>::iterator best_cost = min_element(begin(total_costs), end(total_costs));
    int best_idx = distance(begin(total_costs), best_cost);



    if (best_idx == 0) {      // keep lane
        egoparams.lane = lane;
        egoparams.target_speed = Update_target_speed(vehicle_ahead.dist, vehicle_ahead.speed, target_speed);
        egoparams.ChangeLane = false;
    }

    else if (best_idx == 1) { // change to left
        egoparams.lane = lane - 1 ;
        egoparams.target_speed = Update_target_speed(vehicle_left_ahead.dist, vehicle_left_ahead.speed, target_speed);
        egoparams.ChangeLane = true;
    }

    else {                    // change to right
        egoparams.lane = lane + 1;
        egoparams.target_speed = Update_target_speed(vehicle_right_ahead.dist, vehicle_right_ahead.speed, target_speed);
        egoparams.ChangeLane = true;

    return egoparams;

    }
} 

vector<vector<double>> getTrajectory(const EgoVehicle &egovehicle, const EgoParams &egoparams, const MapWayPoints &map_waypoints, const PreviousPath &previous_path)
{
    // ego vehicle data
    double car_x = egovehicle.x;
    double car_y = egovehicle.y;
    double car_s = egovehicle.s;
    double car_yaw = egovehicle.yaw;
    
    int lane = egoparams.lane;
    double target_speed = egoparams.target_speed;
    bool ChangeLane = egoparams.ChangeLane;

    // map waypoints
    vector<double> map_waypoints_x = map_waypoints.map_waypoints_x;
    vector<double> map_waypoints_y = map_waypoints.map_waypoints_y;
    vector<double> map_waypoints_s = map_waypoints.map_waypoints_s;
  
    // previous path points
    vector<double> previous_path_x = previous_path.previous_path_x;
    vector<double> previous_path_y = previous_path.previous_path_y;

    // trajectory points
    vector<double> next_x_vals;
    vector<double> next_y_vals;

    // ref point as the origin of car/local coordinates
    double ref_x = car_x;
    double ref_y = car_y;
    double ref_yaw = deg2rad(car_yaw);

    // creating spline points
    vector<double> ptsx;
    vector<double> ptsy;
            
    // no previous path , vehicle's starting reference
    int prev_size = previous_path_x.size();
    if (prev_size < 2) { 
        double prev_car_x = car_x - cos(car_yaw);
        double prev_car_y = car_y - sin(car_yaw);

        ptsx.push_back(prev_car_x);
        ptsx.push_back(car_x);

        ptsy.push_back(prev_car_y);
        ptsy.push_back(car_y);
    } 
    // vehicle's previous path ending points
    else {  
        ref_x = previous_path_x[prev_size-1];
        ref_y = previous_path_y[prev_size-1];

        double ref_x_prev = previous_path_x[prev_size-2];
        double ref_y_prev = previous_path_y[prev_size-2];
        ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

        ptsx.push_back(ref_x_prev);
        ptsx.push_back(ref_x);

        ptsy.push_back(ref_y_prev);
        ptsy.push_back(ref_y);
    } 

    // generating points for smooth curve
    vector<double> next_point(2);
    double dist = 30.0;
    if (ChangeLane) { dist = 40;} 
    
    for (int i = 1; i <= 3; ++i) {
        next_point = getXY((car_s + dist * i), (2+4*lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
        ptsx.push_back(next_point[0]);
        ptsy.push_back(next_point[1]);
    } 

    // convertion to vehicle's local coordinates   
    for (int i = 0; i < ptsx.size(); ++i) {      

        double shift_x = ptsx[i] - ref_x;
        double shift_y = ptsy[i] - ref_y;

        ptsx[i] = (shift_x * cos(0-ref_yaw) - shift_y*sin(0-ref_yaw)); 
        ptsy[i] = (shift_x * sin(0-ref_yaw) + shift_y*cos(0-ref_yaw)); 
    }
    
    //  spline
    tk::spline s;
    s.set_points(ptsx, ptsy);
  
    // adding points
    for (int i = 0; i < previous_path_x.size(); i++) {
        next_x_vals.push_back(previous_path_x[i]);
        next_y_vals.push_back(previous_path_y[i]);
    }

    // spitting spline points
    double target_x = 30.0; 
    if (ChangeLane) { target_x = 40;}
    
    double target_y = s(target_x);
    double target_dist = sqrt(target_x * target_x + target_y * target_y);
    
    double x_start = 0;
   
    double N = (target_dist/(0.02*target_speed));
    double offset = (target_x) / N;

    // adding points 
    for (int i = 1; i <= 50-previous_path_x.size(); ++i) {

        double x_point = x_start + offset;
        double y_point = s(x_point);
        x_start = x_point;

        // convertion to global coordinates
        double x_global = ref_x +  (x_point * cos(ref_yaw) - y_point*sin(ref_yaw));
        double y_global = ref_y +  (x_point * sin(ref_yaw) + y_point*cos(ref_yaw));

        next_x_vals.push_back(x_global);
        next_y_vals.push_back(y_global);

    } 
  
    return {next_x_vals, next_y_vals};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

 int lane = 1;
 double target_speed = 0.0;

  h.onMessage([&lane, &target_speed, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];
          
          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;


          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */
         

          int previous_path_size = previous_path_x.size();

          if(previous_path_size > 0)
          {
            car_s = end_path_s;
          }

          EgoVehicle egovehicle;
          egovehicle.x = car_x;
          egovehicle.y = car_y;
          egovehicle.s = car_s;
          egovehicle.d = car_d;
          egovehicle.yaw = car_yaw;
          egovehicle.speed = car_speed;

          MapWayPoints mapwaypts = {map_waypoints_x,map_waypoints_y,map_waypoints_s,map_waypoints_dx,map_waypoints_dy};
          PreviousPath prevpath = {previous_path_x, previous_path_y};

          //prediction
          map<string, VehicleNearby> predictions = Predict(sensor_fusion, car_s, previous_path_size, lane);
          //behaviour planning
          EgoParams egoparams = Plan(predictions, target_speed, lane);
          //trajectory
          auto nextpath = getTrajectory(egovehicle, egoparams, mapwaypts, prevpath);

          lane = egoparams.lane;
          target_speed = egoparams.target_speed;
		
          
          
          msgJson["next_x"] = nextpath[0];
          msgJson["next_y"] = nextpath[1];

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}