import os

game_list = ['alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider',\
  'bowling', 'boxing', 'breakout', 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', 'enduro',\
  'fishing_derby', 'freeway','frostbite','gopher','gravitar','hero','ice_hockey','jamesbond','kangaroo','krull','kung_fu_master',\
  'montezuma_revenge','ms_pacman','name_this_game','pong','private_eye','qbert','riverraid','road_runner','robotank','seaquest',\
  'space_invaders','star_gunner','tennis','time_pilot','tutankham','up_n_down','venture','video_pinball','wizard_of_wor','zaxxon']

  
for game in game_list:
    dir = "LogFiles/" + game
    try:
        os.stat(dir)
    except OSError:
        os.makedirs(dir)