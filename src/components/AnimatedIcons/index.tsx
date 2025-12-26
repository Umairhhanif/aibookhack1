/**
 * Animated Icon Components
 * Reusable SVG icons with beautiful animations for the curriculum
 */

import React from 'react';
import styles from './styles.module.css';

// Robot Icon - Represents ROS 2 / Robotics Fundamentals
export const RobotIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.robotIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="robotGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#3b82f6" />
        <stop offset="100%" stopColor="#8b5cf6" />
      </linearGradient>
    </defs>
    {/* Body */}
    <rect x="16" y="20" width="32" height="28" rx="4" stroke="url(#robotGradient)" strokeWidth="2.5" fill="none" className={styles.robotBody} />
    {/* Eyes */}
    <circle cx="26" cy="32" r="4" fill="url(#robotGradient)" className={styles.robotEye} />
    <circle cx="38" cy="32" r="4" fill="url(#robotGradient)" className={styles.robotEye} />
    {/* Mouth */}
    <rect x="28" y="40" width="8" height="3" rx="1.5" fill="url(#robotGradient)" />
    {/* Antenna */}
    <rect x="30" y="8" width="4" height="12" fill="url(#robotGradient)" />
    <circle cx="32" cy="6" r="4" fill="url(#robotGradient)" className={styles.robotAntenna} />
    {/* Arms */}
    <rect x="6" y="28" width="10" height="12" rx="3" fill="url(#robotGradient)" className={styles.robotLeftArm} />
    <rect x="48" y="28" width="10" height="12" rx="3" fill="url(#robotGradient)" className={styles.robotRightArm} />
    {/* Legs */}
    <rect x="20" y="48" width="8" height="12" rx="2" fill="url(#robotGradient)" />
    <rect x="36" y="48" width="8" height="12" rx="2" fill="url(#robotGradient)" />
  </svg>
);

// Brain Icon - Represents AI / Neural Networks
export const BrainIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.brainIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="brainGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#8b5cf6" />
        <stop offset="100%" stopColor="#ec4899" />
      </linearGradient>
    </defs>
    {/* Brain outline */}
    <path 
      className={styles.brainOutline} 
      d="M32 8C20 8 12 18 12 28C12 38 20 48 32 56C44 48 52 38 52 28C52 18 44 8 32 8Z" 
      stroke="url(#brainGradient)" 
      strokeWidth="2.5" 
      fill="none" 
    />
    {/* Neural connections */}
    <line x1="24" y1="24" x2="40" y2="24" stroke="url(#brainGradient)" strokeWidth="2" className={styles.synapse} />
    <line x1="24" y1="24" x2="32" y2="36" stroke="url(#brainGradient)" strokeWidth="2" className={styles.synapse} />
    <line x1="40" y1="24" x2="32" y2="36" stroke="url(#brainGradient)" strokeWidth="2" className={styles.synapse} />
    <line x1="20" y1="32" x2="32" y2="36" stroke="url(#brainGradient)" strokeWidth="2" className={styles.synapse} />
    <line x1="44" y1="32" x2="32" y2="36" stroke="url(#brainGradient)" strokeWidth="2" className={styles.synapse} />
    {/* Neurons */}
    <circle cx="24" cy="24" r="5" fill="url(#brainGradient)" className={styles.neuron} />
    <circle cx="40" cy="24" r="5" fill="url(#brainGradient)" className={`${styles.neuron} ${styles.neuronDelay1}`} />
    <circle cx="32" cy="36" r="5" fill="url(#brainGradient)" className={`${styles.neuron} ${styles.neuronDelay2}`} />
    <circle cx="20" cy="32" r="4" fill="url(#brainGradient)" className={`${styles.neuron} ${styles.neuronDelay3}`} />
    <circle cx="44" cy="32" r="4" fill="url(#brainGradient)" className={`${styles.neuron} ${styles.neuronDelay1}`} />
  </svg>
);

// Sensor Icon - Represents Digital Twin / Sensors
export const SensorIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.sensorIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="sensorGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#10b981" />
        <stop offset="100%" stopColor="#06b6d4" />
      </linearGradient>
    </defs>
    {/* Center sensor */}
    <circle cx="32" cy="32" r="8" fill="url(#sensorGradient)" className={styles.sensorCore} />
    {/* Wave rings */}
    <circle cx="32" cy="32" r="14" stroke="url(#sensorGradient)" strokeWidth="2" fill="none" className={styles.wave} />
    <circle cx="32" cy="32" r="20" stroke="url(#sensorGradient)" strokeWidth="2" fill="none" className={`${styles.wave} ${styles.waveDelay1}`} />
    <circle cx="32" cy="32" r="26" stroke="url(#sensorGradient)" strokeWidth="2" fill="none" className={`${styles.wave} ${styles.waveDelay2}`} />
  </svg>
);

// Vision Icon - Represents Computer Vision / Perception
export const VisionIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.visionIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="visionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#f97316" />
        <stop offset="100%" stopColor="#ec4899" />
      </linearGradient>
    </defs>
    {/* Eye shape */}
    <ellipse cx="32" cy="32" rx="26" ry="14" stroke="url(#visionGradient)" strokeWidth="2.5" fill="none" className={styles.eyeOutline} />
    {/* Iris */}
    <circle cx="32" cy="32" r="10" stroke="url(#visionGradient)" strokeWidth="2" fill="none" className={styles.iris} />
    {/* Pupil */}
    <circle cx="32" cy="32" r="5" fill="url(#visionGradient)" className={styles.pupil} />
    {/* Scan line */}
    <line x1="6" y1="32" x2="58" y2="32" stroke="url(#visionGradient)" strokeWidth="1.5" strokeDasharray="4 2" className={styles.scanLine} />
  </svg>
);

// Navigation Icon - Represents Nav2 / Path Planning
export const NavigationIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.navigationIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="navGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#06b6d4" />
        <stop offset="100%" stopColor="#3b82f6" />
      </linearGradient>
    </defs>
    {/* Compass circle */}
    <circle cx="32" cy="32" r="26" stroke="url(#navGradient)" strokeWidth="2.5" fill="none" className={styles.compassRing} />
    {/* Direction markers */}
    <line x1="32" y1="8" x2="32" y2="14" stroke="url(#navGradient)" strokeWidth="2" />
    <line x1="32" y1="50" x2="32" y2="56" stroke="url(#navGradient)" strokeWidth="2" />
    <line x1="8" y1="32" x2="14" y2="32" stroke="url(#navGradient)" strokeWidth="2" />
    <line x1="50" y1="32" x2="56" y2="32" stroke="url(#navGradient)" strokeWidth="2" />
    {/* Compass needle */}
    <polygon points="32,16 38,38 32,34 26,38" fill="url(#navGradient)" className={styles.compassNeedle} />
    <polygon points="32,48 38,38 32,42 26,38" fill="#94a3b8" className={styles.compassNeedle} />
  </svg>
);

// Speech Icon - Represents Audio / Speech Integration
export const SpeechIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.speechIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="speechGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#ec4899" />
        <stop offset="100%" stopColor="#f97316" />
      </linearGradient>
    </defs>
    {/* Microphone body */}
    <rect x="24" y="12" width="16" height="24" rx="8" stroke="url(#speechGradient)" strokeWidth="2.5" fill="none" className={styles.micBody} />
    {/* Microphone stand */}
    <path d="M16 32C16 40 23 48 32 48C41 48 48 40 48 32" stroke="url(#speechGradient)" strokeWidth="2.5" fill="none" />
    <line x1="32" y1="48" x2="32" y2="56" stroke="url(#speechGradient)" strokeWidth="2.5" />
    <line x1="24" y1="56" x2="40" y2="56" stroke="url(#speechGradient)" strokeWidth="2.5" strokeLinecap="round" />
    {/* Sound waves */}
    <path d="M52 24C54 28 54 36 52 40" stroke="url(#speechGradient)" strokeWidth="2" strokeLinecap="round" className={styles.soundWave} />
    <path d="M56 20C60 26 60 38 56 44" stroke="url(#speechGradient)" strokeWidth="2" strokeLinecap="round" className={`${styles.soundWave} ${styles.soundWaveDelay}`} />
  </svg>
);

// LLM Icon - Represents Language Models
export const LLMIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.llmIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="llmGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#8b5cf6" />
        <stop offset="100%" stopColor="#06b6d4" />
      </linearGradient>
    </defs>
    {/* Chat bubble */}
    <path 
      d="M12 16C12 12 15 9 19 9H45C49 9 52 12 52 16V36C52 40 49 43 45 43H24L16 51V43H19C15 43 12 40 12 36V16Z" 
      stroke="url(#llmGradient)" 
      strokeWidth="2.5" 
      fill="none" 
      className={styles.chatBubble}
    />
    {/* Text lines */}
    <line x1="20" y1="20" x2="44" y2="20" stroke="url(#llmGradient)" strokeWidth="2" strokeLinecap="round" className={styles.textLine} />
    <line x1="20" y1="26" x2="38" y2="26" stroke="url(#llmGradient)" strokeWidth="2" strokeLinecap="round" className={`${styles.textLine} ${styles.textLineDelay1}`} />
    <line x1="20" y1="32" x2="42" y2="32" stroke="url(#llmGradient)" strokeWidth="2" strokeLinecap="round" className={`${styles.textLine} ${styles.textLineDelay2}`} />
    {/* Sparkle */}
    <circle cx="48" cy="14" r="3" fill="url(#llmGradient)" className={styles.sparkle} />
  </svg>
);

// SLAM Icon - Represents Visual SLAM
export const SLAMIcon = ({ size = 64, className = '' }) => (
  <svg 
    className={`${styles.icon} ${styles.slamIcon} ${className}`} 
    width={size} 
    height={size} 
    viewBox="0 0 64 64" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="slamGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#10b981" />
        <stop offset="100%" stopColor="#8b5cf6" />
      </linearGradient>
    </defs>
    {/* Map grid */}
    <rect x="10" y="10" width="44" height="44" rx="4" stroke="url(#slamGradient)" strokeWidth="2" fill="none" />
    <line x1="10" y1="25" x2="54" y2="25" stroke="url(#slamGradient)" strokeWidth="1" opacity="0.5" />
    <line x1="10" y1="40" x2="54" y2="40" stroke="url(#slamGradient)" strokeWidth="1" opacity="0.5" />
    <line x1="25" y1="10" x2="25" y2="54" stroke="url(#slamGradient)" strokeWidth="1" opacity="0.5" />
    <line x1="40" y1="10" x2="40" y2="54" stroke="url(#slamGradient)" strokeWidth="1" opacity="0.5" />
    {/* Path */}
    <path 
      d="M18 46L25 32L38 36L46 18" 
      stroke="url(#slamGradient)" 
      strokeWidth="2.5" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      fill="none" 
      className={styles.pathLine}
    />
    {/* Waypoints */}
    <circle cx="18" cy="46" r="4" fill="url(#slamGradient)" className={styles.waypoint} />
    <circle cx="25" cy="32" r="3" fill="url(#slamGradient)" className={`${styles.waypoint} ${styles.waypointDelay1}`} />
    <circle cx="38" cy="36" r="3" fill="url(#slamGradient)" className={`${styles.waypoint} ${styles.waypointDelay2}`} />
    <circle cx="46" cy="18" r="4" fill="url(#slamGradient)" className={`${styles.waypoint} ${styles.waypointDelay3}`} />
  </svg>
);

export default {
  RobotIcon,
  BrainIcon,
  SensorIcon,
  VisionIcon,
  NavigationIcon,
  SpeechIcon,
  LLMIcon,
  SLAMIcon,
};
