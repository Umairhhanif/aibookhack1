import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

// Typewriter Effect Component
interface TypewriterProps {
  texts: string[];
  typingSpeed?: number;
  deletingSpeed?: number;
  pauseTime?: number;
}

function Typewriter({ texts, typingSpeed = 80, deletingSpeed = 50, pauseTime = 2000 }: TypewriterProps) {
  const [displayText, setDisplayText] = useState('');
  const [textIndex, setTextIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const currentText = texts[textIndex];
    
    if (isPaused) {
      const pauseTimeout = setTimeout(() => {
        setIsPaused(false);
        setIsDeleting(true);
      }, pauseTime);
      return () => clearTimeout(pauseTimeout);
    }

    if (isDeleting) {
      if (displayText === '') {
        setIsDeleting(false);
        setTextIndex((prev) => (prev + 1) % texts.length);
      } else {
        const deleteTimeout = setTimeout(() => {
          setDisplayText(displayText.slice(0, -1));
        }, deletingSpeed);
        return () => clearTimeout(deleteTimeout);
      }
    } else {
      if (displayText === currentText) {
        setIsPaused(true);
      } else {
        const typeTimeout = setTimeout(() => {
          setDisplayText(currentText.slice(0, displayText.length + 1));
        }, typingSpeed);
        return () => clearTimeout(typeTimeout);
      }
    }
  }, [displayText, textIndex, isDeleting, isPaused, texts, typingSpeed, deletingSpeed, pauseTime]);

  return (
    <span className={styles.typewriter}>
      {displayText}
      <span className={styles.cursor}>|</span>
    </span>
  );
}

// Animated Icon Components
const RobotIcon = () => (
  <svg className={styles.animatedIcon} viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="robotGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#3b82f6" />
        <stop offset="100%" stopColor="#8b5cf6" />
      </linearGradient>
    </defs>
    <rect x="16" y="20" width="32" height="28" rx="4" stroke="url(#robotGrad)" strokeWidth="2" className={styles.robotBody} />
    <circle cx="26" cy="32" r="4" fill="url(#robotGrad)" className={styles.robotEye} />
    <circle cx="38" cy="32" r="4" fill="url(#robotGrad)" className={styles.robotEye} />
    <rect x="28" y="40" width="8" height="4" rx="1" fill="url(#robotGrad)" />
    <rect x="30" y="8" width="4" height="12" fill="url(#robotGrad)" />
    <circle cx="32" cy="6" r="4" fill="url(#robotGrad)" className={styles.robotAntenna} />
    <rect x="8" y="28" width="8" height="12" rx="2" fill="url(#robotGrad)" className={styles.robotArm} />
    <rect x="48" y="28" width="8" height="12" rx="2" fill="url(#robotGrad)" className={styles.robotArm} />
    <rect x="20" y="48" width="8" height="10" rx="2" fill="url(#robotGrad)" />
    <rect x="36" y="48" width="8" height="10" rx="2" fill="url(#robotGrad)" />
  </svg>
);

const BrainIcon = () => (
  <svg className={styles.animatedIcon} viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="brainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#8b5cf6" />
        <stop offset="100%" stopColor="#ec4899" />
      </linearGradient>
    </defs>
    <path className={styles.brainPulse} d="M32 8C20 8 12 18 12 28C12 38 20 48 32 56C44 48 52 38 52 28C52 18 44 8 32 8Z" stroke="url(#brainGrad)" strokeWidth="2" fill="none" />
    <circle cx="24" cy="24" r="6" fill="url(#brainGrad)" className={styles.neuron} />
    <circle cx="40" cy="24" r="6" fill="url(#brainGrad)" className={styles.neuron} style={{ animationDelay: '0.5s' }} />
    <circle cx="32" cy="36" r="6" fill="url(#brainGrad)" className={styles.neuron} style={{ animationDelay: '1s' }} />
    <line x1="24" y1="24" x2="40" y2="24" stroke="url(#brainGrad)" strokeWidth="2" className={styles.synapse} />
    <line x1="24" y1="24" x2="32" y2="36" stroke="url(#brainGrad)" strokeWidth="2" className={styles.synapse} />
    <line x1="40" y1="24" x2="32" y2="36" stroke="url(#brainGrad)" strokeWidth="2" className={styles.synapse} />
  </svg>
);

const SensorIcon = () => (
  <svg className={styles.animatedIcon} viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="sensorGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#10b981" />
        <stop offset="100%" stopColor="#06b6d4" />
      </linearGradient>
    </defs>
    <circle cx="32" cy="32" r="8" fill="url(#sensorGrad)" />
    <circle cx="32" cy="32" r="16" stroke="url(#sensorGrad)" strokeWidth="2" fill="none" className={styles.wave} />
    <circle cx="32" cy="32" r="24" stroke="url(#sensorGrad)" strokeWidth="2" fill="none" className={styles.wave} style={{ animationDelay: '0.5s' }} />
    <circle cx="32" cy="32" r="30" stroke="url(#sensorGrad)" strokeWidth="2" fill="none" className={styles.wave} style={{ animationDelay: '1s' }} />
  </svg>
);

const VisionIcon = () => (
  <svg className={styles.animatedIcon} viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="visionGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#f97316" />
        <stop offset="100%" stopColor="#ec4899" />
      </linearGradient>
    </defs>
    <ellipse cx="32" cy="32" rx="28" ry="16" stroke="url(#visionGrad)" strokeWidth="2" fill="none" />
    <circle cx="32" cy="32" r="10" stroke="url(#visionGrad)" strokeWidth="2" fill="none" />
    <circle cx="32" cy="32" r="4" fill="url(#visionGrad)" className={styles.pupil} />
    <path d="M4 32C4 32 16 16 32 16C48 16 60 32 60 32" stroke="url(#visionGrad)" strokeWidth="2" fill="none" className={styles.eyelid} />
  </svg>
);

// Particle Background Component
function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
    }> = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticles = () => {
      particles = [];
      const count = Math.floor((canvas.width * canvas.height) / 15000);
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          size: Math.random() * 2 + 1,
          opacity: Math.random() * 0.5 + 0.2,
        });
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(59, 130, 246, ${p.opacity})`;
        ctx.fill();

        // Draw connections
        particles.slice(i + 1).forEach((p2) => {
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(59, 130, 246, ${0.2 * (1 - dist / 120)})`;
            ctx.stroke();
          }
        });
      });

      animationId = requestAnimationFrame(animate);
    };

    resize();
    createParticles();
    animate();

    window.addEventListener('resize', () => {
      resize();
      createParticles();
    });

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className={styles.particleCanvas} />;
}

// Feature Card Component
function FeatureCard({ icon, title, description, link, delay }) {
  return (
    <div className={styles.featureCard} style={{ animationDelay: `${delay}ms` }}>
      <div className={styles.featureIcon}>{icon}</div>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
      <Link className={styles.featureLink} to={link}>
        Explore Module →
      </Link>
    </div>
  );
}

// Hero Section
function HeroSection() {
  const { siteConfig } = useDocusaurusContext();
  
  return (
    <header className={styles.hero}>
      <ParticleBackground />
      <div className={styles.heroContent}>
        <div className={styles.heroIconContainer}>
          <RobotIcon />
        </div>
        <h1 className={styles.heroTitle}>
          <span className={styles.gradientText}>Physical AI</span>
          <br />
          <span className={styles.subtitleText}>& Humanoid Robotics</span>
        </h1>
        <p className={styles.heroSubtitle}>
          <Typewriter 
            texts={[
              "Master the future of robotics with hands-on AI integration.",
              "From ROS 2 fundamentals to Vision-Language-Action models.",
              "Build intelligent robots that see, think, and act.",
              "Learn to create digital twins and autonomous navigation.",
              "Integrate LLMs for conversational robot control."
            ]}
            typingSpeed={60}
            deletingSpeed={30}
            pauseTime={2500}
          />
        </p>
        <div className={styles.heroButtons}>
          <Link className={styles.primaryButton} to="/docs/lab-setup/overview">
            <span>🚀</span> Get Started
          </Link>
          <Link className={styles.secondaryButton} to="/docs/module-1-robotic-nervous-system/week-01/introduction">
            <span>📖</span> View Curriculum
          </Link>
        </div>
        <div className={styles.heroStats}>
          <div className={styles.stat}>
            <span className={styles.statNumber}>4</span>
            <span className={styles.statLabel}>Modules</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statNumber}>12</span>
            <span className={styles.statLabel}>Weeks</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statNumber}>50+</span>
            <span className={styles.statLabel}>Exercises</span>
          </div>
        </div>
      </div>
      <div className={styles.scrollIndicator}>
        <div className={styles.scrollArrow}></div>
      </div>
    </header>
  );
}

// Modules Section
function ModulesSection() {
  const modules = [
    {
      icon: <RobotIcon />,
      title: 'Module 1: Robotic Nervous System',
      description: 'Master ROS 2, the middleware powering modern robotics. Learn nodes, topics, services, and actions.',
      link: '/docs/module-1-robotic-nervous-system/week-01/introduction',
      delay: 100,
    },
    {
      icon: <SensorIcon />,
      title: 'Module 2: Digital Twin',
      description: 'Build virtual robot worlds with Gazebo and NVIDIA Isaac Sim for testing and training.',
      link: '/docs/module-2-digital-twin/week-04/introduction',
      delay: 200,
    },
    {
      icon: <BrainIcon />,
      title: 'Module 3: AI Robot Brain',
      description: 'Implement visual SLAM, navigation, and deep learning perception pipelines.',
      link: '/docs/module-3-ai-robot-brain/week-07/introduction',
      delay: 300,
    },
    {
      icon: <VisionIcon />,
      title: 'Module 4: Vision-Language-Action',
      description: 'Integrate LLMs and VLA models for intelligent, conversational robot control.',
      link: '/docs/module-4-vision-language-action/week-10/introduction',
      delay: 400,
    },
  ];

  return (
    <section className={styles.modulesSection}>
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>
          <span className={styles.animatedTitle}>
            {'Curriculum'.split('').map((letter, i) => (
              <span 
                key={i} 
                className={styles.animatedLetter}
                style={{ animationDelay: `${i * 0.1}s` }}
              >
                {letter}
              </span>
            ))}
          </span>
          {' '}
          <span className={styles.gradientText}>Modules</span>
        </h2>
        <p className={styles.sectionSubtitle}>
          <Typewriter 
            texts={[
              "A comprehensive 12-week journey from fundamentals to cutting-edge AI robotics",
              "Master ROS 2, simulation, perception, and intelligent control",
              "Hands-on exercises with real robots and digital twins",
              "From beginner concepts to advanced Vision-Language-Action models"
            ]}
            typingSpeed={40}
            deletingSpeed={20}
            pauseTime={3000}
          />
        </p>
      </div>
      <div className={styles.modulesGrid}>
        {modules.map((module, idx) => (
          <FeatureCard key={idx} {...module} />
        ))}
      </div>
    </section>
  );
}

// Hardware Profiles Section
function HardwareSection() {
  return (
    <section className={styles.hardwareSection}>
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>
          <span className={styles.gradientText}>Choose Your Setup</span>
        </h2>
        <p className={styles.sectionSubtitle}>
          Three hardware profiles to match your resources and learning goals
        </p>
      </div>
      <div className={styles.hardwareGrid}>
        <div className={clsx(styles.hardwareCard, styles.workstation)}>
          <div className={styles.hardwareIcon}>🖥️</div>
          <h3>Digital Twin Workstation</h3>
          <p>Full-power simulation with Isaac Sim, Gazebo, and Unity</p>
          <ul>
            <li>64GB RAM, RTX 4070 Ti+</li>
            <li>Maximum performance</li>
            <li>All features available</li>
          </ul>
          <Link className={styles.hardwareLink} to="/docs/lab-setup/digital-twin-workstation">
            Setup Guide →
          </Link>
        </div>
        <div className={clsx(styles.hardwareCard, styles.jetson)}>
          <div className={styles.hardwareIcon}>🤖</div>
          <h3>Economy Jetson Kit</h3>
          <p>Edge computing with real sensors and deployment</p>
          <ul>
            <li>Jetson Orin Nano</li>
            <li>RealSense D435i</li>
            <li>Hands-on learning</li>
          </ul>
          <Link className={styles.hardwareLink} to="/docs/lab-setup/economy-jetson-kit">
            Setup Guide →
          </Link>
        </div>
        <div className={clsx(styles.hardwareCard, styles.cloud)}>
          <div className={styles.hardwareIcon}>☁️</div>
          <h3>Cloud Ether Lab</h3>
          <p>No local hardware required—run in the cloud</p>
          <ul>
            <li>Browser-based</li>
            <li>Usage-based cost</li>
            <li>Quick start</li>
          </ul>
          <Link className={styles.hardwareLink} to="/docs/lab-setup/cloud-ether-lab">
            Setup Guide →
          </Link>
        </div>
      </div>
    </section>
  );
}

// Main Page Component
export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - Master AI Robotics`}
      description="Comprehensive curriculum for Physical AI and Humanoid Robotics. From ROS 2 fundamentals to Vision-Language-Action models.">
      <main className={styles.main}>
        <HeroSection />
        <ModulesSection />
        <HardwareSection />
      </main>
    </Layout>
  );
}
